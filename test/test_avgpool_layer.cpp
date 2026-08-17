//
// Created by twqb on 8/18/26.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../source/layer/detail/adaptive_avgpooling.hpp"
#include "data/tensor_util.hpp"

using namespace naive_infer;

TEST(AdaptiveAvgPool, ForwardTest) {
  // 假设构造时指定 output_h=2, output_w=2
  std::shared_ptr<AdaptiveAveragePoolingLayer> avg_layer = std::make_shared<AdaptiveAveragePoolingLayer>(2, 2);
  sftensor input = TensorCreate(1,4,4);
  // 填充 4x4 数据
  input->slice(0) = "1,1,2,2;"
                    "1,1,2,2;"
                    "3,3,4,4;"
                    "3,3,4,4;";

  std::vector<sftensor> inputs;
  std::vector<sftensor> outputs;
  inputs.push_back(input);
  outputs.emplace_back(nullptr);

  auto status = avg_layer->Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  auto output = outputs.at(0);
  // 预期输出 2x2：[[1,2],[3,4]]
  LOG(INFO) << "output data: " << output->data();
}