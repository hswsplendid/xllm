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

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace xllm {
namespace {

DSparkBudgetOptions options() {
  DSparkBudgetOptions result;
  result.fixed_draft_time_ms = 1.0;
  return result;
}

const std::vector<DSparkGraphCost> kCosts{{1, 1.0},
                                          {2, 1.0},
                                          {4, 1.1},
                                          {8, 20.0},
                                          {16, 60.0}};

TEST(DSparkGraphControllerTest, BudgetDoesNotReadCurrentOrPreviousConfidence) {
  DSparkGraphController controller(7);
  const std::vector<int32_t> slots{0};
  const std::vector<std::string> ids{"request"};
  const std::vector<double> low(7, 0.2);
  const std::vector<double> high(7, 0.99);
  const auto first = controller.select(slots, ids, low, kCosts, options());
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->budget.draft_budget, 7U);
  ASSERT_TRUE(controller.retire(slots, ids, low, first->budget));
  const auto second = controller.select(slots, ids, high, kCosts, options());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->budget.draft_budget, 7U);
  ASSERT_TRUE(controller.retire(slots, ids, high, second->budget));
  const auto third_low = controller.select(slots, ids, low, kCosts, options());
  const auto third_high =
      controller.select(slots, ids, high, kCosts, options());
  ASSERT_TRUE(third_low.has_value());
  ASSERT_TRUE(third_high.has_value());
  EXPECT_EQ(third_low->budget.draft_budget, third_high->budget.draft_budget);
  EXPECT_LT(third_low->budget.draft_budget, 7U);
  EXPECT_EQ(controller.generation(), 2U);
}

TEST(DSparkGraphControllerTest, CurrentConfidenceOnlyRedistributesFixedBudget) {
  DSparkGraphController controller(7);
  const std::vector<int32_t> slots{0, 1};
  const std::vector<std::string> ids{"a", "b"};
  const std::vector<double> history(14, 0.4);
  for (uint32_t round = 0; round < 2; ++round) {
    const auto plan = controller.select(slots, ids, history, kCosts, options());
    ASSERT_TRUE(plan.has_value());
    ASSERT_TRUE(controller.retire(slots, ids, history, plan->budget));
  }
  std::vector<double> current(14, 0.01);
  std::fill_n(current.begin(), 7, 0.99);
  const auto first = controller.select(slots, ids, current, kCosts, options());
  std::reverse(current.begin(), current.end());
  const auto second = controller.select(slots, ids, current, kCosts, options());
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(first->budget.draft_budget, second->budget.draft_budget);
  EXPECT_EQ(first->prefix_lengths[0], second->prefix_lengths[1]);
  EXPECT_GT(first->prefix_lengths[0], first->prefix_lengths[1]);
  EXPECT_EQ(std::accumulate(
                first->prefix_lengths.begin(), first->prefix_lengths.end(), 0U),
            first->budget.draft_budget);
}

TEST(DSparkGraphControllerTest, SlotReuseAndAdmissionRequireFreshHistory) {
  DSparkGraphController controller(7);
  std::vector<int32_t> slots{3};
  std::vector<std::string> ids{"old"};
  const std::vector<double> confidence(7, 0.2);
  for (uint32_t round = 0; round < 2; ++round) {
    const auto plan =
        controller.select(slots, ids, confidence, kCosts, options());
    ASSERT_TRUE(plan.has_value());
    ASSERT_TRUE(controller.retire(slots, ids, confidence, plan->budget));
  }
  ids[0] = "new";
  const auto reused =
      controller.select(slots, ids, confidence, kCosts, options());
  ASSERT_TRUE(reused.has_value());
  EXPECT_EQ(reused->budget.draft_budget, 7U);
  ASSERT_TRUE(controller.retire(slots, ids, confidence, reused->budget));
  slots[0] = 4;
  ids[0] = "other";
  const auto admitted =
      controller.select(slots, ids, confidence, kCosts, options());
  ASSERT_TRUE(admitted.has_value());
  ASSERT_TRUE(controller.retire(slots, ids, confidence, admitted->budget));
  EXPECT_EQ(controller.history_size(), 1U);
  slots[0] = 3;
  ids[0] = "new";
  EXPECT_EQ(controller.select(slots, ids, confidence, kCosts, options())
                ->budget.draft_budget,
            7U);
}

TEST(DSparkGraphControllerTest, RequestReorderingPreservesIdentity) {
  DSparkGraphController controller(7);
  std::vector<int32_t> slots{4, 9};
  std::vector<std::string> ids{"a", "b"};
  const std::vector<double> confidence(14, 0.3);
  for (uint32_t round = 0; round < 2; ++round) {
    const auto plan =
        controller.select(slots, ids, confidence, kCosts, options());
    ASSERT_TRUE(plan.has_value());
    ASSERT_TRUE(controller.retire(slots, ids, confidence, plan->budget));
  }
  const auto before =
      controller.select(slots, ids, confidence, kCosts, options());
  std::reverse(slots.begin(), slots.end());
  std::reverse(ids.begin(), ids.end());
  const auto after =
      controller.select(slots, ids, confidence, kCosts, options());
  ASSERT_TRUE(before.has_value());
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(before->budget.draft_budget, after->budget.draft_budget);
  EXPECT_LT(after->budget.draft_budget, 14U);
}

TEST(DSparkGraphControllerTest, InvalidInputsDoNotRetireOrMutateHistory) {
  DSparkGraphController controller(7);
  const std::vector<int32_t> slots{0};
  const std::vector<std::string> ids{"a"};
  std::vector<double> confidence(7, 0.5);
  confidence[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
      controller.select(slots, ids, confidence, kCosts, options()).has_value());
  EXPECT_FALSE(controller.retire(slots, ids, confidence, {7, 8}));
  EXPECT_EQ(controller.generation(), 0U);
  EXPECT_EQ(controller.history_size(), 0U);
  confidence[1] = 0.5;
  EXPECT_FALSE(controller.retire(slots, ids, confidence, {8, 16}));
  EXPECT_FALSE(controller.retire(slots, ids, confidence, {7, 4}));
  const std::vector<DSparkGraphCost> insufficient_capacity{{1, 1.0}};
  EXPECT_FALSE(
      controller
          .select(slots, ids, confidence, insufficient_capacity, options())
          .has_value());
}

TEST(DSparkGraphControllerTest, BinaryConfidenceTiesKeepExactPrefixBudget) {
  for (double value : {0.0, 1.0}) {
    DSparkGraphController controller(7);
    const std::vector<int32_t> slots{0, 1};
    const std::vector<std::string> ids{"a", "b"};
    const std::vector<double> confidence(14, value);
    for (uint32_t round = 0; round < 5; ++round) {
      const auto plan =
          controller.select(slots, ids, confidence, kCosts, options());
      ASSERT_TRUE(plan.has_value());
      EXPECT_EQ(
          std::accumulate(
              plan->prefix_lengths.begin(), plan->prefix_lengths.end(), 0U),
          plan->budget.draft_budget);
      for (int32_t length : plan->prefix_lengths) {
        EXPECT_GE(length, 0);
        EXPECT_LE(length, 7);
      }
      ASSERT_TRUE(controller.retire(slots, ids, confidence, plan->budget));
    }
  }
}

TEST(DSparkGraphControllerTest,
     RetiredAllocationMatchesExplicitPreviousConfidence) {
  DSparkGraphController controller(7);
  const std::vector<int32_t> slots{0, 1};
  const std::vector<std::string> ids{"a", "b"};
  std::vector<double> confidence(14, 0.2);
  std::fill_n(confidence.begin(), 7, 0.95);
  for (uint32_t round = 0; round < 2; ++round) {
    const auto plan = controller.select(slots, ids, {}, kCosts, options());
    ASSERT_TRUE(plan.has_value());
    ASSERT_TRUE(controller.retire(slots, ids, confidence, plan->budget));
  }
  const auto retired = controller.select(slots, ids, {}, kCosts, options());
  const auto explicit_previous =
      controller.select(slots, ids, confidence, kCosts, options());
  ASSERT_TRUE(retired.has_value());
  ASSERT_TRUE(explicit_previous.has_value());
  EXPECT_EQ(retired->prefix_lengths, explicit_previous->prefix_lengths);
  EXPECT_EQ(retired->budget.draft_budget,
            explicit_previous->budget.draft_budget);
}

TEST(DSparkGraphControllerTest, NeverSelectsUnreachableCheapLargerBucket) {
  DSparkGraphController controller(7);
  const std::vector<int32_t> slots{0};
  const std::vector<std::string> ids{"a"};
  const std::vector<double> confidence(7, 0.99);
  const std::vector<DSparkGraphCost> costs{
      {1, 1.0}, {2, 2.0}, {4, 3.0}, {8, 4.0}, {16, 0.001}};
  for (uint32_t round = 0; round < 4; ++round) {
    const auto plan =
        controller.select(slots, ids, confidence, costs, options());
    ASSERT_TRUE(plan.has_value());
    EXPECT_LE(plan->budget.graph_rows, 8U);
    ASSERT_TRUE(controller.retire(slots, ids, confidence, plan->budget));
  }
}

TEST(DSparkGraphControllerTest, MatchesExhaustiveTwoRequestPrefixOracle) {
  for (uint32_t width : {5U, 7U}) {
    const std::vector<DSparkGraphCost> costs{
        {1, 0.9}, {2, 1.0}, {4, 1.3}, {8, 3.4}, {16, 8.0}};
    for (double first : {0.0, 0.2, 0.7, 1.0}) {
      for (double second : {0.0, 0.3, 0.9, 1.0}) {
        DSparkGraphController controller(width);
        const std::vector<int32_t> slots{0, 1};
        const std::vector<std::string> ids{"a", "b"};
        std::vector<double> confidence(width * 2, second);
        std::fill_n(confidence.begin(), width, first);
        DSparkBudgetOptions settings = options();
        settings.draft_width = width;
        for (uint32_t round = 0; round < 2; ++round) {
          const auto plan =
              controller.select(slots, ids, confidence, costs, settings);
          ASSERT_TRUE(plan.has_value());
          ASSERT_TRUE(controller.retire(slots, ids, confidence, plan->budget));
        }
        const auto plan =
            controller.select(slots, ids, confidence, costs, settings);
        ASSERT_TRUE(plan.has_value());
        const auto score = [&](uint32_t left, uint32_t right) {
          double progress = 2.0;
          double survival = 1.0;
          for (uint32_t index = 0; index < left; ++index) {
            survival *= first;
            progress += survival;
          }
          survival = 1.0;
          for (uint32_t index = 0; index < right; ++index) {
            survival *= second;
            progress += survival;
          }
          const auto cost = std::find_if(
              costs.begin(), costs.end(), [&](const DSparkGraphCost& value) {
                return value.graph_rows >= 2 + left + right;
              });
          return progress /
                 (settings.fixed_draft_time_ms + cost->target_time_ms);
        };
        double oracle = 0.0;
        for (uint32_t left = 0; left <= width; ++left) {
          for (uint32_t right = 0; right <= width; ++right) {
            oracle = std::max(oracle, score(left, right));
          }
        }
        EXPECT_NEAR(score(plan->prefix_lengths[0], plan->prefix_lengths[1]),
                    oracle,
                    1e-12);
      }
    }
  }
}

}  // namespace
}  // namespace xllm
