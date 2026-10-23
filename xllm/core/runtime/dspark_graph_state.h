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

#include <c10/core/Event.h>
#include <torch/types.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/framework/speculative/dspark/dspark_graph_controller.h"
#include "core/framework/speculative/dspark/dspark_runtime_cost_model.h"

namespace xllm {

// CUDA single-worker integration. Timing events are read at the verification
// synchronization that already exists; confidence transfer is asynchronous.
class DSparkGraphState final {
 public:
  DSparkGraphState(std::vector<DSparkGraphCost> costs,
                   uint32_t draft_width,
                   uint32_t max_requests,
                   double min_gain,
                   bool enable_feedback,
                   bool shadow);

  std::vector<int32_t> begin(std::span<const int32_t> slots,
                             std::span<const std::string> request_ids,
                             uint32_t kv_length,
                             const c10::Stream& stream);
  void begin_verify(const c10::Stream& stream);
  void end_verify(const c10::Stream& stream);
  void stage_confidence(const torch::Tensor& confidence);
  void finish(const c10::Stream& stream);

 private:
  struct Timing {
    c10::Event start{c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT};
    c10::Event verify_start{c10::DeviceType::CUDA,
                            c10::EventFlag::BACKEND_DEFAULT};
    c10::Event verify_end{c10::DeviceType::CUDA,
                          c10::EventFlag::BACKEND_DEFAULT};
    c10::Event end{c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT};
    uint32_t batch_size = 0;
    uint32_t kv_length = 0;
    uint32_t graph_rows = 0;
    uint64_t generation = 0;
  };

  DSparkGraphController controller_;
  DSparkRuntimeCostModel cost_model_;
  DSparkBudgetOptions options_;
  bool shadow_ = false;
  bool active_ = false;
  bool confidence_staged_ = false;
  uint32_t max_requests_ = 0;
  std::array<Timing, 2> timings_;
  std::vector<int32_t> slots_;
  std::vector<std::string> request_ids_;
  DSparkBudgetPlan executed_plan_;
  torch::Tensor confidence_host_;
  torch::Tensor confidence_device_;
};

}  // namespace xllm
