//
// Created by twqb on 8/15/26.
//
#include <gtest/gtest.h>
#include "data/tensor.hpp"
#include <glog/logging.h>

using namespace naive_infer;

TEST(test_tensor, create_tensor) {
  Tensor<float> t1(2,3,4);
  Tensor<float> t2(3,4);
  Tensor<float> t3(4);
  LOG(INFO) << "t1 shapes size: " << t1.raw_shapes().size();
  LOG(INFO) << "t2 shapes size: " << t2.raw_shapes().size();
  LOG(INFO) << "t3 shapes size: " << t3.raw_shapes().size();
}