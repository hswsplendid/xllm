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

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/speculative/dspark/dspark_budget_planner.h"

namespace xllm {

struct DSparkGraphDecision {
  DSparkBudgetPlan budget;
  std::vector<int32_t> prefix_lengths;
};

// Single-worker state. A budget sees only retired confidence from generation
// n-2; current confidence may redistribute that fixed budget, never resize it.
class DSparkGraphController final {
 public:
  explicit DSparkGraphController(uint32_t draft_width);

  // Empty current_confidence uses n-1 retired feedback for allocation, with
  // equal confidence for newly admitted requests. No Device read is needed.
  std::optional<DSparkGraphDecision> select(
      std::span<const int32_t> slots,
      std::span<const std::string> request_ids,
      std::span<const double> current_confidence,
      std::span<const DSparkGraphCost> costs,
      const DSparkBudgetOptions& options);

  // Called only after successful verification. Invalid input does not change
  // either history or the generation visible to subsequent decisions.
  bool retire(std::span<const int32_t> slots,
              std::span<const std::string> request_ids,
              std::span<const double> confidence,
              const DSparkBudgetPlan& executed_plan);

  uint64_t generation() const;
  size_t history_size() const;

 private:
  struct Feedback {
    uint64_t generation = 0;
    std::array<double, 7> confidence{};
  };
  struct History {
    std::string request_id;
    std::array<Feedback, 2> feedback;
    uint32_t samples = 0;
  };

  bool valid_input(std::span<const int32_t> slots,
                   std::span<const std::string> request_ids,
                   std::span<const double> confidence) const;

  uint32_t draft_width_ = 0;
  uint64_t generation_ = 0;
  uint64_t rounds_since_full_width_ = 0;
  std::unordered_map<int32_t, History> history_;
  std::vector<int32_t> previous_slots_;
  std::vector<std::string> previous_request_ids_;
  std::optional<DSparkBudgetPlan> previous_plan_;
};

}  // namespace xllm
