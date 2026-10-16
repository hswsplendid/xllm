include(cc_test)

cc_test(
  NAME
    xattention_test
  SRCS
    xattention_test.cpp
  DEPS
    :config
    :cuda_layers
    :common_layers
    torch
    GTest::gtest_main
    glog::glog
    :block
)
# Let CMake scan the full static-library cycle, including runtime and models.
target_link_libraries(xattention_test PRIVATE brpc)

cc_test(
  NAME
    rwkv7_decoder_layer_test
  SRCS
    rwkv7_decoder_layer_test.cpp
  DEPS
    :config
    :layers
    :common_layers
    :kv_cache
    :model_context
    :state_dict
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(rwkv7_decoder_layer_test PRIVATE brpc)
