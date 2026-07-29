//
// Created by wjldw on 2026/7/30.
//
#include "tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor, tensor_init1D) {
    using namespace naive_infer;
    Tensor<float> f1(4);
    f1.Fill(1.f);
    const auto &raw_shapes = f1.raw_shapes();
    LOG(INFO) << "-----------------------Tensor1D-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t size = raw_shapes.at(0);
    LOG(INFO) << "data numbers: " << size;
    f1.Show();
}