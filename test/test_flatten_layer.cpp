//
// Created by twqb on 8/20/26.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../source/layer/detail/flatten.hpp"
#include "data/tensor.hpp"

using namespace naive_infer;

// 基础正向测试：start_dim=1 end_dim=3，把 [C,H,W] 展平成一维
TEST(TestFlattenLayer, ForwardBasicCase) {
  // 构造输入：1个batch，Tensor维度 C=2, H=3, W=3
  auto input_tensor = std::make_shared<Tensor<float>>(2, 3, 3);
  std::vector<float> data{
    1.f, 2.f, 3.f,
    4.f, 5.f, 6.f,
    7.f, 8.f, 9.f,

    10.f,11.f,12.f,
    13.f,14.f,15.f,
    16.f,17.f,18.f
};
  input_tensor->Fill(data);

  std::vector<std::shared_ptr<Tensor<float>>> inputs{input_tensor};
  std::vector<std::shared_ptr<Tensor<float>>> outputs{nullptr};

  // 构造Flatten：start_dim=1 end_dim=3，对应合并 C H W
  FlattenLayer layer(1, 3);
  InferStatus status = layer.Forward(inputs, outputs);

  // 校验推理成功
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  auto out = outputs.at(0);

  // 2*3*3 = 18 个元素，展平为一维
  ASSERT_EQ(out->size(), 18);
  ASSERT_EQ(out->channels(), 1);
  ASSERT_EQ(out->rows(), 1);
  ASSERT_EQ(out->cols(), 18);

  LOG(INFO) << "output data: "<< out->data();
}