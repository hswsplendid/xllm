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

// Build with the three dspark planner/controller/cost-model .cpp files; no
// Torch or device runtime is needed. CSV stdout is the benchmark artifact.
// The CMake target additionally measures the original Torch controller on
// host and CUDA inputs, including its synchronous confidence transfer.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include "core/framework/speculative/dspark/dspark_graph_controller.h"
#include "core/framework/speculative/dspark/dspark_runtime_cost_model.h"

#if defined(XLLM_BENCHMARK_TORCH)
#include <torch/cuda.h>
#include <torch/torch.h>

#include "core/framework/speculative/adaptive_speculative_controller.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#endif

namespace {
thread_local uint64_t allocation_calls = 0;
thread_local bool count_allocations = false;
}  // namespace

void* operator new(size_t size) {
  void* memory = std::malloc(size == 0 ? 1 : size);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  if (count_allocations) {
    ++allocation_calls;
  }
  return memory;
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, size_t) noexcept { std::free(memory); }
void* operator new[](size_t size) { return ::operator new(size); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory, size_t) noexcept { std::free(memory); }

namespace {

uint64_t checksum = 0;
std::ofstream raw_samples;

void measure(const std::string& name,
             uint32_t batch_size,
             uint32_t width,
             const std::function<void()>& operation) {
  constexpr uint32_t kWarmup = 100;
  constexpr uint32_t kSamples = 200;
  constexpr uint32_t kIterations = 50;
  for (uint32_t index = 0; index < kWarmup; ++index) {
    operation();
  }
  std::vector<double> samples;
  samples.reserve(kSamples);
  for (uint32_t sample = 0; sample < kSamples; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
      operation();
    }
    const auto end = std::chrono::steady_clock::now();
    samples.emplace_back(
        std::chrono::duration<double, std::nano>(end - start).count() /
        kIterations);
  }
  if (raw_samples.is_open()) {
    for (uint32_t index = 0; index < kSamples; ++index) {
      raw_samples << name << ',' << batch_size << ',' << width << ',' << index
                  << ',' << samples[index] << '\n';
    }
  }
  std::sort(samples.begin(), samples.end());
  allocation_calls = 0;
  count_allocations = true;
  operation();
  count_allocations = false;
  std::cout << name << ',' << batch_size << ',' << width << ','
            << samples[kSamples / 2] << ',' << samples[kSamples * 95 / 100]
            << ',' << allocation_calls << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc > 2) {
    std::cerr << "Usage: dspark_graph_policy_benchmark [raw-samples.csv]\n";
    return 2;
  }
  if (argc == 2) {
    raw_samples.open(argv[1]);
    if (!raw_samples) {
      std::cerr << "Cannot open raw samples output\n";
      return 2;
    }
    raw_samples << "operation,batch_size,draft_width,sample,ns\n";
  }
#if defined(XLLM_BENCHMARK_TORCH)
  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  xllm::SpeculativeProfileRegistry::get_instance().set_validate_time_predictor(
      {.intercept_ms = 1.0, .query_token_ms = 0.03, .query_prefix_ms = 0.0});
#endif
  std::cout
      << "operation,batch_size,draft_width,median_ns,p95_ns,allocations\n";
  for (uint32_t width : {5U, 7U}) {
    for (uint32_t batch : {1U, 4U, 16U, 32U, 64U, 128U}) {
      std::vector<int32_t> slots(batch);
      std::iota(slots.begin(), slots.end(), 0);
      std::vector<std::string> ids;
      ids.reserve(batch);
      for (uint32_t index = 0; index < batch; ++index) {
        ids.emplace_back("request-" + std::to_string(index));
      }
      std::vector<double> confidence(batch * width);
      for (size_t index = 0; index < confidence.size(); ++index) {
        confidence[index] = 0.2 + (index * 37 % 79) * 0.01;
      }
      std::vector<xllm::DSparkGraphCost> costs;
      costs.reserve(68);
      for (uint32_t rows : {1U, 2U, 4U, 8U}) {
        costs.emplace_back(xllm::DSparkGraphCost{rows, 1.0 + rows * 0.03});
      }
      for (uint32_t rows = 16; rows <= 1024; rows += 16) {
        costs.emplace_back(xllm::DSparkGraphCost{rows, 1.0 + rows * 0.03});
      }
      xllm::DSparkBudgetOptions options;
      options.draft_width = width;
      options.fixed_draft_time_ms = 1.0;
#if defined(XLLM_BENCHMARK_TORCH)
      xllm::runtime::Options legacy_options;
      legacy_options.enable_adaptive_speculative_decode(true)
          .num_speculative_tokens(static_cast<int32_t>(width))
          .speculative_algorithm("DSpark");
      xllm::AdaptiveSpeculativeController legacy(legacy_options);
      const std::vector<double> kv_lengths(batch, 1024.0);
      const torch::Tensor host_confidence =
          torch::from_blob(confidence.data(),
                           {batch, width},
                           torch::TensorOptions().dtype(torch::kFloat64))
              .to(torch::kFloat32);
      measure("legacy_linear_host", batch, width, [&] {
        checksum += legacy.select_pruned_prefix_lengths(
            host_confidence, 1.0, kv_lengths)[0];
      });
      if (torch::cuda::is_available()) {
        const torch::Tensor device_confidence =
            host_confidence.to(torch::kCUDA);
        torch::cuda::synchronize();
        measure("legacy_linear_cuda_transfer", batch, width, [&] {
          checksum += legacy.select_pruned_prefix_lengths(
              device_confidence, 1.0, kv_lengths)[0];
        });
      }
#endif
      measure("existing_budget_planner", batch, width, [&] {
        const auto plan =
            xllm::DSparkBudgetPlanner::plan(batch, confidence, costs, options);
        checksum += plan.value().draft_budget;
      });
      xllm::DSparkGraphController controller(width);
      measure("history_select_and_retire", batch, width, [&] {
        const auto plan = controller.select(slots, ids, {}, costs, options);
        checksum += plan.value().budget.draft_budget;
        if (!controller.retire(slots, ids, confidence, plan->budget)) {
          std::terminate();
        }
      });
      xllm::DSparkRuntimeCostModel model(costs, true);
      uint64_t generation = 0;
      const uint32_t full_rows = batch * (width + 1);
      const auto full = std::find_if(
          costs.begin(), costs.end(), [&](const xllm::DSparkGraphCost& cost) {
            return cost.graph_rows >= full_rows;
          });
      measure("feedback_observe_and_estimate", batch, width, [&] {
        if (!model.observe(batch,
                           1024,
                           full->graph_rows,
                           full->target_time_ms + 1.0,
                           2.0,
                           generation++)) {
          std::terminate();
        }
        checksum += model.estimate(batch, 1024, generation).graphs.size();
      });
      xllm::DSparkGraphController combined_controller(width);
      xllm::DSparkRuntimeCostModel combined_model(costs, true);
      measure("graph_policy_round", batch, width, [&] {
        const uint64_t round = combined_controller.generation();
        const xllm::DSparkRuntimeCosts estimate =
            combined_model.estimate(batch, 1024, round);
        options.fixed_draft_time_ms = estimate.fixed_time_ms;
        const auto plan = combined_controller.select(
            slots, ids, {}, estimate.graphs, options);
        checksum += plan.value().budget.draft_budget;
        if (!combined_controller.retire(slots, ids, confidence, plan->budget) ||
            !combined_model.observe(batch,
                                    1024,
                                    plan->budget.graph_rows,
                                    2.0 + plan->budget.graph_rows * 0.03,
                                    1.0,
                                    round)) {
          std::terminate();
        }
      });
    }
  }
  return checksum == 0 ? 1 : 0;
}
