//
// Created by twqb on 8/18/26.
//
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "data/tensor_util.hpp"
#include "../source/layer/detail/relu.hpp"

using namespace naive_infer;

TEST(test_relu_layer, relu_foward) {
  auto relu_layer = std::make_shared<ReluLayer>();
  sftensor input = TensorCreate(2,2,3);
  input->slice(0) = "1,2,3;"
                  "4,5,6;";

  input->slice(1) = "-1,-2,-3;"
                    "-4,-5,-6;";
  LOG(INFO) << "input tensor data: " << input->data();

  sftensor input2 = TensorCreate(2,2,3);
  std::vector<float> data1 = {
    1,2,3,
    4,5,6
  };
  std::vector<float> data2 = {
    -1,-2,-3,
    -4,-5,-6
  };
  float* ptr = input2->raw_ptr();
  // 拷贝通道0
  memcpy(ptr, data1.data(), data1.size() * sizeof(float));
  // 偏移6个float，拷贝通道1
  memcpy(ptr + data1.size(), data2.data(), data2.size() * sizeof(float));
  LOG(INFO) << "input tensor2 data: " << input2->data();

  sftensor input3 = TensorCreate(2,2,3);
  std::vector<float> data = {
    // chan0
    1,2,3,
    4,5,6,
    // chan1
    -1,-2,-3,
    -4,-5,-6
};
  input3->Fill(data, false);
  LOG(INFO) << "input tensor3 data: " << input3->data();

  std::vector<sftensor> inputs;
  std::vector<sftensor> outputs;

  inputs.push_back(input);
  outputs.emplace_back(nullptr);

  InferStatus status = relu_layer->Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  auto& out_tensor = outputs.at(0);
  ASSERT_NE(out_tensor, nullptr);
  LOG(INFO) << "output tensor data: " << out_tensor->data();
}