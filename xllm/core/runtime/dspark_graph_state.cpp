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

#include "core/runtime/dspark_graph_state.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

namespace xllm {

DSparkGraphState::DSparkGraphState(std::vector<DSparkGraphCost> costs,
                                   uint32_t draft_width,
                                   uint32_t max_requests,
                                   double min_gain,
                                   bool enable_feedback,
                                   bool shadow)
    : controller_(draft_width),
      cost_model_(std::move(costs), enable_feedback),
      shadow_(shadow),
      max_requests_(max_requests) {
  CHECK(draft_width == 5 || draft_width == 7);
  CHECK_GT(max_requests, 0U);
  CHECK_LE(max_requests, 128U);
  options_.draft_width = draft_width;
  options_.min_gain = min_gain;
  slots_.reserve(max_requests);
  request_ids_.reserve(max_requests);
  confidence_host_ =
      torch::empty({static_cast<int64_t>(max_requests) * draft_width},
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCPU)
                       .pinned_memory(true));
}

std::vector<int32_t> DSparkGraphState::begin(
    std::span<const int32_t> slots,
    std::span<const std::string> request_ids,
    uint32_t kv_length,
    const c10::Stream& stream) {
  CHECK(!active_);
  CHECK_LE(slots.size(), max_requests_);
  const uint64_t generation = controller_.generation();
  const uint32_t batch_size = static_cast<uint32_t>(slots.size());
  const DSparkRuntimeCosts costs =
      cost_model_.estimate(batch_size, kv_length, generation);
  options_.fixed_draft_time_ms = costs.fixed_time_ms;
  const auto decision =
      controller_.select(slots, request_ids, {}, costs.graphs, options_);
  CHECK(decision.has_value()) << "No valid warmed DSpark graph plan";
  executed_plan_ = decision->budget;
  std::vector<int32_t> prefixes = decision->prefix_lengths;
  if (shadow_) {
    executed_plan_.draft_budget = batch_size * options_.draft_width;
    const uint32_t rows = batch_size + executed_plan_.draft_budget;
    const auto full =
        std::lower_bound(costs.graphs.begin(),
                         costs.graphs.end(),
                         rows,
                         [](const DSparkGraphCost& cost, uint32_t requested) {
                           return cost.graph_rows < requested;
                         });
    CHECK(full != costs.graphs.end());
    executed_plan_.graph_rows = full->graph_rows;
    prefixes.assign(batch_size, static_cast<int32_t>(options_.draft_width));
  }
  slots_.assign(slots.begin(), slots.end());
  request_ids_.assign(request_ids.begin(), request_ids.end());
  Timing& timing = timings_[generation % timings_.size()];
  timing.batch_size = batch_size;
  timing.kv_length = kv_length;
  timing.graph_rows = executed_plan_.graph_rows;
  timing.generation = generation;
  timing.start.record(stream);
  active_ = true;
  confidence_staged_ = false;
  VLOG(1) << "DSpark graph plan generation=" << generation
          << " requests=" << batch_size << " kv=" << kv_length
          << " budget=" << executed_plan_.draft_budget
          << " graph=" << executed_plan_.graph_rows
          << " proposed_graph=" << decision->budget.graph_rows
          << " fixed_ms=" << costs.fixed_time_ms;
  return prefixes;
}

void DSparkGraphState::begin_verify(const c10::Stream& stream) {
  CHECK(active_);
  timings_[controller_.generation() % timings_.size()].verify_start.record(
      stream);
}

void DSparkGraphState::end_verify(const c10::Stream& stream) {
  CHECK(active_);
  timings_[controller_.generation() % timings_.size()].verify_end.record(
      stream);
}

void DSparkGraphState::stage_confidence(const torch::Tensor& confidence) {
  CHECK(active_ && !confidence_staged_);
  CHECK(confidence.defined() && confidence.is_cuda());
  CHECK_EQ(confidence.numel(), slots_.size() * options_.draft_width);
  // Retain the source until the existing compute-stream synchronization.
  confidence_device_ = confidence.to(torch::kFloat32).contiguous().view({-1});
  confidence_host_.narrow(0, 0, confidence_device_.numel())
      .copy_(confidence_device_, true);
  confidence_staged_ = true;
}

void DSparkGraphState::finish(const c10::Stream& stream) {
  CHECK(active_ && confidence_staged_);
  const uint64_t generation = controller_.generation();
  Timing& current = timings_[generation % timings_.size()];
  CHECK(current.verify_end.query())
      << "DSpark feedback must retire after verification sync";
  // The previous commit runs before the current round on the same stream.
  // Its timing is therefore complete without any new synchronization.
  if (generation > 0) {
    Timing& previous = timings_[(generation - 1) % timings_.size()];
    CHECK(previous.end.query());
    const double verify_ms =
        previous.verify_start.elapsedTime(previous.verify_end);
    const double total_ms = previous.start.elapsedTime(previous.end);
    CHECK(cost_model_.observe(previous.batch_size,
                              previous.kv_length,
                              previous.graph_rows,
                              verify_ms,
                              total_ms - verify_ms,
                              previous.generation))
        << "Invalid completed DSpark graph timing";
    VLOG(1) << "DSpark graph timing generation=" << previous.generation
            << " verify_ms=" << verify_ms
            << " fixed_ms=" << total_ms - verify_ms;
  }
  std::vector<double> confidence;
  confidence.reserve(slots_.size() * options_.draft_width);
  const float* values = confidence_host_.data_ptr<float>();
  for (size_t index = 0; index < slots_.size() * options_.draft_width;
       ++index) {
    confidence.emplace_back(values[index]);
  }
  CHECK(controller_.retire(slots_, request_ids_, confidence, executed_plan_));
  confidence_device_ = torch::Tensor();
  current.end.record(stream);
  active_ = false;
}

}  // namespace xllm
