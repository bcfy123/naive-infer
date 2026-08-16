//
// Created by twqb on 8/16/26.
//
#include <gtest/gtest.h>
#include "data/tensor_util.hpp"
#include <glog/logging.h>

using namespace naive_infer;

TEST(test_tensor_util, test_broadcast) {
  sftensor t1 = TensorCreate(2,3,4);
  sftensor t2 = TensorCreate(2,1,1);
  std::vector<float> values(t1->size());
  std::iota(values.begin(), values.end(), 1.f);
  t1->Fill(values);
  t2->Fill(1.f);
  LOG(INFO) << "t1 data: " << t1->data();
  LOG(INFO) << "t2 data: " << t2->data();
  const auto& [out_t1, out_t2] = TensorBroadcast(t1, t2);
  LOG(INFO) << "out_t1 data: " << out_t1->data();
  LOG(INFO) << "out_t2 data: " << out_t2->data();
}

TEST(test_tensor_util, test_elementadd) {
  sftensor t1 = TensorCreate(2,3,4);
  sftensor t2 = TensorCreate(2,1,1);
  std::vector<float> values(t1->size());
  std::iota(values.begin(), values.end(), 1.f);
  t1->Fill(values);
  t2->Fill(2.f);
  LOG(INFO) << "t1 data: " << t1->data();
  LOG(INFO) << "t2 data: " << t2->data();
  auto add_t = TensorElementAdd(t1, t2);
  LOG(INFO) << "add data: " << add_t->data();
  auto multiply_t = TensorElementMultiply(t1,t2);
  LOG(INFO) << "multiply data: " << multiply_t->data();
}