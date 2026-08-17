//
// Created by twqb on 8/17/26.
//
#include <gtest/gtest.h>
#include "data/tensor.hpp"
#include <vector>
#include <memory>
#include "../source/layer/detail/linear.hpp"
#include "layer/abstract/layer_factory.hpp"

using namespace naive_infer;

TEST(test_linear_layer, ForwardNoBias) {
  // 超参：in=2, out=3, feature_dims=2（2个token）
  int32_t in_features = 2;
  int32_t out_features = 3;
  int32_t feature_dims = 2;
  bool use_bias = false;

  auto linear = std::make_shared<LinearLayer>(in_features, out_features, use_bias);

  // 2. 设置权重 W: [out, in] = 3x2
  std::vector<float> weight_data = {
      1.f, 2.f,
      3.f, 4.f,
      5.f, 6.f
  };
  linear->set_weights(weight_data);

  // 3. 构造输入tensor：shape [1, feature_dims=2, in_features=2]
  // X = [[10,20],
  //      [30,40]]
  std::vector<float> input_data = {
      10.f, 20.f,
      30.f, 40.f
  };
  auto input_tensor = std::make_shared<Tensor<float>>(1, feature_dims, in_features);
  memcpy(input_tensor->raw_ptr(), input_data.data(), input_data.size() * sizeof(float));

  // 4. 准备输入输出vector（batch=1，只1个tensor）
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input_tensor);
  outputs.emplace_back(nullptr); // 交给算子内部自动分配

  // 5. 跑Forward
  InferStatus status = linear->Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  auto& out_tensor = outputs.at(0);
  ASSERT_NE(out_tensor, nullptr);

  // 6. 手动计算真值 Y = X @ W.T
  // W.T = [[1,3,5],
  //        [2,4,6]]
  // X[0] = [10,20] → [10*1+20*2, 10*3+20*4, 10*5+20*6] = [50,110,170]
  // X[1] = [30,40] → [30*1+40*2, 30*3+40*4, 30*5+40*6] = [110,250,390]
  std::vector<float> expect = {
      50.f, 110.f, 170.f,
      110.f,250.f,390.f
  };
  LOG(INFO) << "output tensor data: " << out_tensor->data();
}

// case2: 带bias Linear
TEST(test_linear_layer, ForwardWithBias) {
  int32_t in_features = 2;
  int32_t out_features = 3;
  int32_t feature_dims = 2;
  bool use_bias = true;

  auto linear = std::make_shared<LinearLayer>(in_features, out_features, use_bias);

  // W 和上面一样
  std::vector<float> weight_data = {
      1.f, 2.f,
      3.f, 4.f,
      5.f, 6.f
  };
  linear->set_weights(weight_data);
  // bias [1,1,1]
  std::vector<float> bias_data = {1.f,1.f,1.f};
  linear->set_bias(bias_data);

  // 输入X不变
  std::vector<float> input_data = {
      10.f, 20.f,
      30.f, 40.f
  };
  auto input_tensor = std::make_shared<Tensor<float>>(1, feature_dims, in_features);
  memcpy(input_tensor->raw_ptr(), input_data.data(), input_data.size() * sizeof(float));

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input_tensor);
  outputs.emplace_back(nullptr);

  InferStatus status = linear->Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  auto& out_tensor = outputs.at(0);

  // 真值 = 上面结果 +1
  std::vector<float> expect = {
      51.f, 111.f, 171.f,
      111.f,251.f,391.f
  };
  float* out_ptr = out_tensor->raw_ptr();
  for (size_t i=0; i<expect.size(); i++) {
    ASSERT_NEAR(out_ptr[i], expect[i], 1e-5f);
  }
}