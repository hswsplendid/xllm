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

#include "core/framework/speculative/dspark/dspark_runtime_cost_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xllm {
namespace {

constexpr size_t kMinimumSamples = 8;
constexpr size_t kWindowSize = 31;
constexpr uint64_t kMaximumSampleAge = 256;
constexpr uint32_t kMaximumBatchSize = 128;

uint64_t window_key(uint32_t batch_size, uint32_t kv_length) {
  return (static_cast<uint64_t>(batch_size) << 32) |
         DSparkRuntimeCostModel::context_bucket(kv_length);
}

double median(std::vector<double> values) {
  const auto middle = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), middle, values.end());
  if (values.size() % 2 != 0) {
    return *middle;
  }
  return (*middle + *std::max_element(values.begin(), middle)) * 0.5;
}

}  // namespace

DSparkRuntimeCostModel::DSparkRuntimeCostModel(
    std::vector<DSparkGraphCost> startup_costs,
    bool enable_feedback)
    : startup_costs_(std::move(startup_costs)),
      enable_feedback_(enable_feedback) {}

uint32_t DSparkRuntimeCostModel::context_bucket(uint32_t kv_length) {
  // A finite 128 x 24 table bounds serving memory. Saturation also avoids an
  // overflow for a caller that supplies a length near UINT32_MAX.
  uint32_t bucket = 256;
  while (bucket < kv_length &&
         bucket <= std::numeric_limits<uint32_t>::max() / 2) {
    bucket *= 2;
  }
  return bucket;
}

bool DSparkRuntimeCostModel::observe(uint32_t batch_size,
                                     uint32_t kv_length,
                                     uint32_t graph_rows,
                                     double verify_time_ms,
                                     double fixed_time_ms,
                                     uint64_t generation) {
  if (batch_size == 0 || batch_size > kMaximumBatchSize ||
      graph_rows < batch_size || !std::isfinite(verify_time_ms) ||
      verify_time_ms <= 0.0 || !std::isfinite(fixed_time_ms) ||
      fixed_time_ms <= 0.0) {
    return false;
  }
  const auto baseline =
      std::lower_bound(startup_costs_.begin(),
                       startup_costs_.end(),
                       graph_rows,
                       [](const DSparkGraphCost& cost, uint32_t rows) {
                         return cost.graph_rows < rows;
                       });
  if (baseline == startup_costs_.end() || baseline->graph_rows != graph_rows ||
      !std::isfinite(baseline->target_time_ms) ||
      baseline->target_time_ms <= 0.0) {
    return false;
  }
  Window& window = windows_[window_key(batch_size, kv_length)];
  Window& shared = fixed_windows_[context_bucket(kv_length)];
  if ((!window.samples.empty() && generation <= window.last_generation) ||
      (!shared.samples.empty() && generation <= shared.last_generation)) {
    return false;
  }
  // Frozen calibration and online feedback use identical first eight samples.
  // This makes their performance difference attributable to subsequent updates.
  const auto append = [&](Window& destination, double slope) {
    destination.last_generation = generation;
    if (!enable_feedback_ && destination.samples.size() >= kMinimumSamples) {
      return;
    }
    destination.samples.emplace_back(Sample{slope, fixed_time_ms, generation});
    while (destination.samples.size() > kWindowSize) {
      destination.samples.pop_front();
    }
  };
  const double slope =
      std::max(0.0, (verify_time_ms - baseline->target_time_ms) / graph_rows);
  append(window, slope);
  append(shared, /*slope=*/0.0);
  return true;
}

DSparkRuntimeCosts DSparkRuntimeCostModel::estimate(uint32_t batch_size,
                                                    uint32_t kv_length,
                                                    uint64_t generation) const {
  DSparkRuntimeCosts result{startup_costs_, 0.0};
  if (batch_size == 0 || batch_size > kMaximumBatchSize) {
    return result;
  }
  const auto found = windows_.find(window_key(batch_size, kv_length));
  std::vector<double> slopes;
  std::vector<double> fixed_times;
  slopes.reserve(kWindowSize);
  fixed_times.reserve(kWindowSize);
  const auto eligible = [&](const Sample& sample) {
    return sample.generation < generation &&
           (!enable_feedback_ ||
            generation - sample.generation <= kMaximumSampleAge);
  };
  if (found != windows_.end()) {
    for (const Sample& sample : found->second.samples) {
      if (!eligible(sample)) {
        continue;
      }
      slopes.emplace_back(sample.slope_ms);
      fixed_times.emplace_back(sample.fixed_time_ms);
    }
  }
  if (slopes.size() < kMinimumSamples) {
    // A new batch size may disappear before accumulating eight rounds. Use
    // completed context-local fixed costs until its own calibration is ready;
    // never borrow another batch size's verification slope.
    fixed_times.clear();
    const auto shared = fixed_windows_.find(context_bucket(kv_length));
    if (shared == fixed_windows_.end()) {
      return result;
    }
    for (const Sample& sample : shared->second.samples) {
      if (!eligible(sample)) {
        continue;
      }
      fixed_times.emplace_back(sample.fixed_time_ms);
    }
    if (fixed_times.size() >= kMinimumSamples) {
      result.fixed_time_ms = median(std::move(fixed_times));
    }
    return result;
  }
  const double slope = median(std::move(slopes));
  result.fixed_time_ms = median(std::move(fixed_times));
  for (DSparkGraphCost& graph : result.graphs) {
    graph.target_time_ms += slope * graph.graph_rows;
  }
  return result;
}

}  // namespace xllm
