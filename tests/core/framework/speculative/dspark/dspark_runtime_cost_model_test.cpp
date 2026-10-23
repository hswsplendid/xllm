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

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace xllm {
namespace {

const std::vector<DSparkGraphCost> kCosts{{1, 1.0},
                                          {2, 1.5},
                                          {4, 2.0},
                                          {8, 4.0}};

TEST(DSparkRuntimeCostModelTest,
     RequiresRetiredMinimumSamplesAndIsolatesVerificationGeometry) {
  DSparkRuntimeCostModel model(kCosts, true);
  for (uint64_t generation = 0; generation < 8; ++generation) {
    EXPECT_DOUBLE_EQ(model.estimate(1, 128, generation).fixed_time_ms, 0.0);
    ASSERT_TRUE(model.observe(1, 128, 8, 8.0, 3.0, generation));
  }
  EXPECT_DOUBLE_EQ(model.estimate(1, 128, 7).fixed_time_ms, 0.0);
  const auto costs = model.estimate(1, 128, 8);
  EXPECT_DOUBLE_EQ(costs.fixed_time_ms, 3.0);
  EXPECT_DOUBLE_EQ(costs.graphs[2].target_time_ms, 4.0);
  EXPECT_DOUBLE_EQ(model.estimate(2, 128, 8).fixed_time_ms, 3.0);
  EXPECT_DOUBLE_EQ(model.estimate(2, 128, 8).graphs[2].target_time_ms, 2.0);
  EXPECT_DOUBLE_EQ(model.estimate(1, 257, 8).fixed_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(model.estimate(1, 256, 8).fixed_time_ms, 3.0);
}

TEST(DSparkRuntimeCostModelTest,
     SharedPriorSurvivesBatchChurnUntilOwnWindowReady) {
  DSparkRuntimeCostModel model(kCosts, true);
  for (uint64_t generation = 0; generation < 8; ++generation) {
    EXPECT_DOUBLE_EQ(model.estimate(3, 128, generation).fixed_time_ms, 0.0);
    ASSERT_TRUE(
        model.observe(generation % 2 + 1, 128, 8, 8.0, 3.0, generation));
  }
  EXPECT_DOUBLE_EQ(model.estimate(3, 128, 7).fixed_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(model.estimate(3, 128, 8).fixed_time_ms, 3.0);
  EXPECT_DOUBLE_EQ(model.estimate(3, 128, 8).graphs[3].target_time_ms, 4.0);
  EXPECT_FALSE(model.observe(3, 128, 8, 80.0, 30.0, 7));
  for (uint64_t generation = 8; generation < 16; ++generation) {
    ASSERT_TRUE(model.observe(3, 128, 8, 12.0, 6.0, generation));
  }
  const auto costs = model.estimate(3, 128, 16);
  EXPECT_DOUBLE_EQ(costs.fixed_time_ms, 6.0);
  EXPECT_DOUBLE_EQ(costs.graphs[3].target_time_ms, 12.0);
  EXPECT_DOUBLE_EQ(model.estimate(4, 128, 16).fixed_time_ms, 4.5);
  EXPECT_DOUBLE_EQ(model.estimate(3, 257, 16).fixed_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(model.estimate(0, 128, 16).fixed_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(model.estimate(129, 128, 16).fixed_time_ms, 0.0);
}

TEST(DSparkRuntimeCostModelTest, MedianRejectsOutliersAndExpiredFeedback) {
  DSparkRuntimeCostModel model(kCosts, true);
  for (uint64_t generation = 0; generation < 10; ++generation) {
    ASSERT_TRUE(model.observe(1,
                              128,
                              8,
                              generation == 3 ? 8000.0 : 8.0,
                              generation == 3 ? 3000.0 : 3.0,
                              generation));
  }
  const auto costs = model.estimate(1, 128, 10);
  EXPECT_DOUBLE_EQ(costs.fixed_time_ms, 3.0);
  EXPECT_DOUBLE_EQ(costs.graphs[3].target_time_ms, 8.0);
  EXPECT_DOUBLE_EQ(model.estimate(1, 128, 300).fixed_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(model.estimate(2, 128, 10).fixed_time_ms, 3.0);
  EXPECT_DOUBLE_EQ(model.estimate(2, 128, 300).fixed_time_ms, 0.0);
}

TEST(DSparkRuntimeCostModelTest, InvalidSamplesDoNotPoisonWindow) {
  DSparkRuntimeCostModel model(kCosts, true);
  EXPECT_FALSE(model.observe(1, 128, 3, 4.0, 3.0, 0));
  EXPECT_FALSE(model.observe(0, 128, 8, 4.0, 3.0, 0));
  EXPECT_FALSE(model.observe(129, 128, 256, 4.0, 3.0, 0));
  EXPECT_FALSE(model.observe(1, 128, 8, -1.0, 3.0, 0));
  EXPECT_FALSE(model.observe(
      1, 128, 8, 4.0, std::numeric_limits<double>::infinity(), 0));
  ASSERT_TRUE(model.observe(1, 128, 8, 8.0, 3.0, 1));
  EXPECT_FALSE(model.observe(1, 128, 8, 800.0, 30.0, 1));
  EXPECT_FALSE(model.observe(1, 128, 8, 800.0, 30.0, 0));
  for (uint64_t generation = 2; generation <= 8; ++generation) {
    ASSERT_TRUE(model.observe(1, 128, 8, 8.0, 3.0, generation));
  }
  EXPECT_DOUBLE_EQ(model.estimate(1, 128, 9).fixed_time_ms, 3.0);
}

TEST(DSparkRuntimeCostModelTest,
     FrozenCalibrationAndOnlineFeedbackSeparateCleanly) {
  DSparkRuntimeCostModel frozen(kCosts, false);
  DSparkRuntimeCostModel online(kCosts, true);
  for (uint64_t generation = 0; generation < 48; ++generation) {
    const double verify = generation < 8 ? 8.0 : 12.0;
    const double fixed = generation < 8 ? 3.0 : 6.0;
    ASSERT_TRUE(frozen.observe(1, 128, 8, verify, fixed, generation));
    ASSERT_TRUE(online.observe(1, 128, 8, verify, fixed, generation));
  }
  EXPECT_DOUBLE_EQ(frozen.estimate(1, 128, 48).fixed_time_ms, 3.0);
  EXPECT_DOUBLE_EQ(online.estimate(1, 128, 48).fixed_time_ms, 6.0);
  EXPECT_DOUBLE_EQ(frozen.estimate(2, 128, 48).fixed_time_ms, 3.0);
  EXPECT_DOUBLE_EQ(online.estimate(2, 128, 48).fixed_time_ms, 6.0);
  EXPECT_DOUBLE_EQ(online.estimate(1, 128, 48).graphs[3].target_time_ms, 12.0);
  EXPECT_DOUBLE_EQ(frozen.estimate(1, 128, 1000).fixed_time_ms, 3.0);
  EXPECT_FALSE(frozen.observe(1, 128, 8, 12.0, 6.0, 47));
  EXPECT_FALSE(frozen.observe(1, 128, 8, 12.0, 6.0, 46));
}

TEST(DSparkRuntimeCostModelTest, NegativeResidualCannotReduceStartupCost) {
  DSparkRuntimeCostModel model(kCosts, true);
  for (uint64_t generation = 0; generation < 8; ++generation) {
    ASSERT_TRUE(model.observe(1, 1, 8, 1.0, 2.0, generation));
  }
  EXPECT_DOUBLE_EQ(model.estimate(1, 1, 8).graphs[3].target_time_ms, 4.0);
  EXPECT_EQ(DSparkRuntimeCostModel::context_bucket(0), 256U);
  EXPECT_EQ(DSparkRuntimeCostModel::context_bucket(257), 512U);
  EXPECT_GT(DSparkRuntimeCostModel::context_bucket(
                std::numeric_limits<uint32_t>::max()),
            0U);
}

}  // namespace
}  // namespace xllm
