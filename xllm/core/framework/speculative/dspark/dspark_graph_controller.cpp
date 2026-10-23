/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/speculative/dspark/dspark_graph_controller.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xllm {
namespace {

struct Candidate {
  uint32_t request = 0;
  uint32_t position = 0;
  double survival = 0.0;
};

}  // namespace

DSparkGraphController::DSparkGraphController(uint32_t draft_width)
    : draft_width_(draft_width) {}

bool DSparkGraphController::valid_input(
    std::span<const int32_t> slots,
    std::span<const std::string> request_ids,
    std::span<const double> confidence) const {
  if ((draft_width_ != 5 && draft_width_ != 7) || slots.empty() ||
      slots.size() != request_ids.size() ||
      slots.size() >
          std::numeric_limits<uint32_t>::max() / (draft_width_ + 1) ||
      confidence.size() != slots.size() * draft_width_) {
    return false;
  }
  std::vector<int32_t> unique_slots(slots.begin(), slots.end());
  std::sort(unique_slots.begin(), unique_slots.end());
  if (unique_slots.front() < 0 ||
      std::adjacent_find(unique_slots.begin(), unique_slots.end()) !=
          unique_slots.end()) {
    return false;
  }
  for (size_t row = 0; row < slots.size(); ++row) {
    if (request_ids[row].empty()) {
      return false;
    }
  }
  return std::all_of(confidence.begin(), confidence.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
  });
}

std::optional<DSparkGraphDecision> DSparkGraphController::select(
    std::span<const int32_t> slots,
    std::span<const std::string> request_ids,
    std::span<const double> current_confidence,
    std::span<const DSparkGraphCost> costs,
    const DSparkBudgetOptions& options) {
  if ((draft_width_ != 5 && draft_width_ != 7) || slots.empty() ||
      slots.size() != request_ids.size() ||
      slots.size() >
          std::numeric_limits<uint32_t>::max() / (draft_width_ + 1)) {
    return std::nullopt;
  }
  std::vector<double> allocation_confidence;
  if (current_confidence.empty()) {
    allocation_confidence.assign(slots.size() * draft_width_, 1.0);
    for (size_t row = 0; row < slots.size(); ++row) {
      const auto found = history_.find(slots[row]);
      if (found == history_.end() ||
          found->second.request_id != request_ids[row] ||
          found->second.samples == 0 || generation_ == 0 ||
          found->second.feedback.back().generation != generation_ - 1) {
        continue;
      }
      const auto& confidence = found->second.feedback.back().confidence;
      std::copy_n(confidence.begin(),
                  draft_width_,
                  allocation_confidence.begin() + row * draft_width_);
    }
    current_confidence = allocation_confidence;
  }
  if (options.draft_width != draft_width_ ||
      !valid_input(slots, request_ids, current_confidence)) {
    return std::nullopt;
  }
  std::vector<double> lagged;
  lagged.reserve(current_confidence.size());
  for (size_t row = 0; row < slots.size() && generation_ >= 2; ++row) {
    const auto found = history_.find(slots[row]);
    if (found == history_.end() ||
        found->second.request_id != request_ids[row] ||
        found->second.samples < 2) {
      break;
    }
    const auto& feedback = found->second.feedback;
    const auto eligible = std::find_if(
        feedback.rbegin(), feedback.rend(), [&](const Feedback& sample) {
          return sample.generation == generation_ - 2;
        });
    if (eligible == feedback.rend()) {
      break;
    }
    lagged.insert(lagged.end(),
                  eligible->confidence.begin(),
                  eligible->confidence.begin() + draft_width_);
  }
  if (lagged.size() != current_confidence.size()) {
    lagged.clear();
  }
  const bool same_requests = std::equal(slots.begin(),
                                        slots.end(),
                                        previous_slots_.begin(),
                                        previous_slots_.end()) &&
                             std::equal(request_ids.begin(),
                                        request_ids.end(),
                                        previous_request_ids_.begin(),
                                        previous_request_ids_.end());
  // Timing noise must not select a bucket above the reachable full-width
  // bucket: allocation cannot force the executor to use that larger graph.
  if (!std::is_sorted(
          costs.begin(),
          costs.end(),
          [](const DSparkGraphCost& lhs, const DSparkGraphCost& rhs) {
            return lhs.graph_rows < rhs.graph_rows;
          })) {
    return std::nullopt;
  }
  const auto full =
      std::lower_bound(costs.begin(),
                       costs.end(),
                       slots.size() * (draft_width_ + 1),
                       [](const DSparkGraphCost& cost, size_t rows) {
                         return cost.graph_rows < rows;
                       });
  if (full == costs.end()) {
    return std::nullopt;
  }
  costs = costs.first(static_cast<size_t>(full - costs.begin()) + 1);
  const auto plan =
      DSparkBudgetPlanner::plan(static_cast<uint32_t>(slots.size()),
                                lagged,
                                costs,
                                options,
                                same_requests ? previous_plan_ : std::nullopt,
                                same_requests ? rounds_since_full_width_ : 0);
  if (!plan.has_value()) {
    return std::nullopt;
  }
  std::vector<Candidate> candidates;
  candidates.reserve(current_confidence.size());
  for (uint32_t row = 0; row < slots.size(); ++row) {
    double survival = 1.0;
    for (uint32_t position = 0; position < draft_width_; ++position) {
      survival *= current_confidence[row * draft_width_ + position];
      candidates.emplace_back(Candidate{row, position, survival});
    }
  }
  // A predecessor always precedes its successor, including confidence 0/1.
  // This makes top-D an exact prefix allocation without reading future tokens
  // in the budget decision. Ties are deterministic across repeated runs.
  std::sort(candidates.begin(),
            candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
              if (lhs.survival != rhs.survival) {
                return lhs.survival > rhs.survival;
              }
              if (lhs.position != rhs.position) {
                return lhs.position < rhs.position;
              }
              return lhs.request < rhs.request;
            });
  DSparkGraphDecision decision{*plan, std::vector<int32_t>(slots.size(), 0)};
  for (uint32_t index = 0; index < plan->draft_budget; ++index) {
    ++decision.prefix_lengths[candidates[index].request];
  }
  return decision;
}

bool DSparkGraphController::retire(std::span<const int32_t> slots,
                                   std::span<const std::string> request_ids,
                                   std::span<const double> confidence,
                                   const DSparkBudgetPlan& executed_plan) {
  if (!valid_input(slots, request_ids, confidence) ||
      executed_plan.draft_budget > slots.size() * draft_width_ ||
      executed_plan.graph_rows < slots.size() + executed_plan.draft_budget ||
      generation_ == std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  history_.reserve(slots.size());
  for (size_t row = 0; row < slots.size(); ++row) {
    History& history = history_[slots[row]];
    if (history.request_id != request_ids[row]) {
      history.request_id = request_ids[row];
      history.samples = 0;
    }
    history.feedback[0] = history.feedback[1];
    history.feedback[1].generation = generation_;
    std::copy_n(confidence.begin() + row * draft_width_,
                draft_width_,
                history.feedback[1].confidence.begin());
    history.samples = std::min(history.samples + 1, 2U);
  }
  std::erase_if(history_, [&](const auto& entry) {
    return entry.second.feedback[1].generation != generation_;
  });
  previous_slots_.assign(slots.begin(), slots.end());
  previous_request_ids_.assign(request_ids.begin(), request_ids.end());
  previous_plan_ = executed_plan;
  rounds_since_full_width_ =
      executed_plan.draft_budget == slots.size() * draft_width_
          ? 0
          : rounds_since_full_width_ + 1;
  ++generation_;
  return true;
}

uint64_t DSparkGraphController::generation() const { return generation_; }
size_t DSparkGraphController::history_size() const { return history_.size(); }

}  // namespace xllm
