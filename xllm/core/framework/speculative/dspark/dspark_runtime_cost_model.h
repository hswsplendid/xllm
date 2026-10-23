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

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

#include "core/framework/speculative/dspark/dspark_budget_planner.h"

namespace xllm {

struct DSparkRuntimeCosts {
  std::vector<DSparkGraphCost> graphs;
  double fixed_time_ms = 0.0;
};

// Worker-owned measurements: no cross-model singleton and no queued-plan
// mutation. Observations contain target verification and the remaining round
// exactly once. Invalid or out-of-order samples leave the model unchanged.
class DSparkRuntimeCostModel final {
 public:
  explicit DSparkRuntimeCostModel(std::vector<DSparkGraphCost> startup_costs,
                                  bool enable_feedback);

  bool observe(uint32_t batch_size,
               uint32_t kv_length,
               uint32_t graph_rows,
               double verify_time_ms,
               double fixed_time_ms,
               uint64_t generation);

  DSparkRuntimeCosts estimate(uint32_t batch_size,
                              uint32_t kv_length,
                              uint64_t generation) const;

  static uint32_t context_bucket(uint32_t kv_length);

 private:
  struct Sample {
    double slope_ms = 0.0;
    double fixed_time_ms = 0.0;
    uint64_t generation = 0;
  };
  struct Window {
    std::deque<Sample> samples;
    uint64_t last_generation = 0;
  };

  std::vector<DSparkGraphCost> startup_costs_;
  bool enable_feedback_ = false;
  std::unordered_map<uint64_t, Window> windows_;
  // A context-local prior for newly encountered batch sizes. Verification
  // residuals remain isolated by both batch size and context length.
  std::unordered_map<uint32_t, Window> fixed_windows_;
};

}  // namespace xllm
