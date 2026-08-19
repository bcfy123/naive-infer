//
// Created by twqb on 8/20/26.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../source/layer/detail/maxpooling.hpp"
#include "data/tensor.hpp"
#include <vector>

using namespace naive_infer;

// 测试1: 基础用例 2x2 maxpool stride=2 padding=0
TEST(TestMaxPooling, Basic2x2Pool) {
    // input: 1 batch, 1 channel, 4x4
    auto input = std::make_shared<Tensor<float>>(1, 4, 4);
    std::vector<float> input_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16
    };
    input->Fill(input_data);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    // output 先空，Forward内部创建
    outputs.emplace_back(nullptr);

    // pool_size=2, stride=2, padding=0
    MaxPoolingLayer layer(0,0,2,2,2,2);
    InferStatus status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    auto out = outputs.at(0);
    ASSERT_EQ(out->channels(), 1);
    ASSERT_EQ(out->rows(), 2);
    ASSERT_EQ(out->cols(), 2);

    // 预期输出
    // [6,8]
    // [14,16]
    const auto& out_mat = out->slice(0);
    LOG(INFO) << "output data: " << out_mat;
}

// 测试2: 带padding
TEST(TestMaxPooling, WithPadding) {
    // input 1ch,3x3
    auto input = std::make_shared<Tensor<float>>(1,3,3);
    std::vector<float> in_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    input->Fill(in_data);

    std::vector<std::shared_ptr<Tensor<float>>> inputs{input};
    std::vector<std::shared_ptr<Tensor<float>>> outputs{nullptr};

    // pool=3, stride=1, padding=1
    MaxPoolingLayer layer(1,1,3,3,1,1);
    auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    auto out = outputs[0];
    ASSERT_EQ(out->rows(),3);
    ASSERT_EQ(out->cols(),3);
    // 中心位置取全部9个数最大值9
    auto& om = out->slice(0);
    EXPECT_FLOAT_EQ(om(1,1),9.f);
}