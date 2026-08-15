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

TEST(test_tensor, tensor_data) {
  Tensor<float> t1(2,3);
  t1.Fill(1);
  LOG(INFO) << "t1 data; " << t1.data();
  std::vector<float> value{1,2,3,4,5,6};
  t1.Fill(value);
  LOG(INFO) << "t1 data; " << t1.data();
  t1.Fill(value, false);
  LOG(INFO) << "t1 data; " << t1.data();
  LOG(INFO) << "t1 size; " << t1.size();
  LOG(INFO) << "t1 index 4; " << t1.index(4);
  LOG(INFO) << "t1 slice 0; " << t1.slice(0);
  arma::fmat& t = t1.slice(0);
  t[1] = 0;
  LOG(INFO) << "t1 slice 0; " << t1.slice(0);
  auto* ptr = t1.raw_ptr();
  LOG(INFO) << "t1 index 0; " << *(ptr+1);
  LOG(INFO) << "t1 index 0; " << *(t1.raw_ptr(2));
  auto* matrix_raw_ptr = t1.matrix_raw_ptr(0);
  LOG(INFO) << "t1 index 0; " << *(matrix_raw_ptr);
}