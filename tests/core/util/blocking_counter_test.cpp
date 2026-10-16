/* Copyright 2025-2026 The xLLM Authors.

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

#include "blocking_counter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>

#include "util/threadpool.h"

namespace xllm {

TEST(BlockingCounterTest, BasicTest) {
  BlockingCounter counter(/*initial_count=*/1);
  counter.decrement_count();
  counter.wait();
  EXPECT_TRUE(true);
}

TEST(BlockingCounterTest, TwoThreadTest) {
  ThreadPool threadpool(/*num_threads=*/1);
  BlockingCounter counter(/*initial_count=*/2);

  // Publish each result before signaling completion; increments may overlap.
  std::atomic<int32_t> called{0};
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  called.fetch_add(1, std::memory_order_relaxed);
  counter.decrement_count();
  counter.wait();
  EXPECT_EQ(called.load(std::memory_order_relaxed), 2);
}

TEST(BlockingCounterTest, MultiThreadTest) {
  ThreadPool threadpool(/*num_threads=*/4);
  BlockingCounter counter(/*initial_count=*/5);

  // Publish each result before signaling completion; increments may overlap.
  std::atomic<int32_t> called{0};
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  called.fetch_add(1, std::memory_order_relaxed);
  counter.decrement_count();

  counter.wait();
  EXPECT_EQ(called.load(std::memory_order_relaxed), 5);
}

TEST(BlockingCounterTest, WaitTimeoutTest) {
  ThreadPool threadpool(/*num_threads=*/2);
  BlockingCounter counter(/*initial_count=*/3);
  BlockingCounter completed(/*initial_count=*/1);

  // Publish each result before signaling completion; increments may overlap.
  std::atomic<int32_t> called{0};
  threadpool.schedule([&counter, &completed, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
    completed.decrement_count();
  });

  called.fetch_add(1, std::memory_order_relaxed);
  counter.decrement_count();

  const std::chrono::milliseconds timeout(100);
  // The worker has finished, but the intentionally unmatched count remains.
  completed.wait();
  EXPECT_FALSE(counter.wait_for(timeout));
  EXPECT_EQ(called.load(std::memory_order_relaxed), 2);
}

}  // namespace xllm
