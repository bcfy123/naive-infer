//
// Created by twqb on 8/18/26.
//

#include <gtest/gtest.h>
#include "../source/layer/detail/cat.hpp"
#include "data/tensor.hpp"
#include "layer/abstract/layer.hpp"

using namespace naive_infer;

TEST(test_cat_layer, ForwardSingleOutputDim1) {
    // case1: 单输出，2个输入张量拼接
    // input0: C=2, H=2, W=2
    auto t0 = std::make_shared<Tensor<float>>(2, 2, 2);
    t0->Fill(1.0f);
    // input1: C=2, H=2, W=2
    auto t1 = std::make_shared<Tensor<float>>(2, 2, 2);
    t1->Fill(2.0f);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(t0);
    inputs.push_back(t1);

    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(nullptr); // output由层内部new

    CatLayer cat_layer(1);
    auto status = cat_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    // 校验输出 shape [4, 2, 2]
    auto out = outputs.at(0);
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(out->channels(), 4);
    ASSERT_EQ(out->rows(), 2);
    ASSERT_EQ(out->cols(), 2);

    LOG(INFO) << "cat layer output: " << out->data();
}

TEST(test_cat_layer, ForwardMultiOutputGroup) {
    // 多输出分组场景，完全匹配代码的j += output_size逻辑
    // outputs.size() =2, inputs.size()=4 → packet_size = 4/2 =2
    // group0(out0): inputs[0], inputs[2]
    // group1(out1): inputs[1], inputs[3]
    auto t0 = std::make_shared<Tensor<float>>(1,1,1); t0->Fill(10.f);
    auto t1 = std::make_shared<Tensor<float>>(1,1,1); t1->Fill(20.f);
    auto t2 = std::make_shared<Tensor<float>>(1,1,1); t2->Fill(11.f);
    auto t3 = std::make_shared<Tensor<float>>(1,1,1); t3->Fill(21.f);

    std::vector<std::shared_ptr<Tensor<float>>> inputs = {t0, t1, t2, t3};
    std::vector<std::shared_ptr<Tensor<float>>> outputs = {nullptr, nullptr};

    CatLayer cat(1);
    auto status = cat.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    auto out0 = outputs[0];
    auto out1 = outputs[1];
    ASSERT_EQ(out0->channels(), 2);
    ASSERT_EQ(out1->channels(), 2);

    // out0 = cat(t0, t2): [10, 11]
    EXPECT_FLOAT_EQ(out0->index(0), 10.f);
    EXPECT_FLOAT_EQ(out0->index(1), 11.f);

    // out1 = cat(t1, t3): [20, 21]
    EXPECT_FLOAT_EQ(out1->index(0), 20.f);
    EXPECT_FLOAT_EQ(out1->index(1), 21.f);
}

TEST(test_cat_layer, ForwardDimNeg3) {
    // dim=-3 和 dim=1等价
    auto a = std::make_shared<Tensor<float>>(2, 1,1); a->Fill(5.f);
    auto b = std::make_shared<Tensor<float>>(3,1,1); b->Fill(6.f);
    std::vector<std::shared_ptr<Tensor<float>>> inputs = {a,b};
    std::vector<std::shared_ptr<Tensor<float>>> outputs = {nullptr};

    CatLayer cat(-3);
    auto st = cat.Forward(inputs, outputs);
    ASSERT_EQ(st, InferStatus::kInferSuccess);
    auto out = outputs[0];
    ASSERT_EQ(out->channels(), 5);
}

TEST(test_cat_layer, ForwardBadDimReturnError) {
    // dim=2，非法维度，应该返回错误
    auto a = std::make_shared<Tensor<float>>(1,1,1);
    std::vector<std::shared_ptr<Tensor<float>>> inputs = {a,a};
    std::vector<std::shared_ptr<Tensor<float>>> outputs = {nullptr};

    CatLayer cat(2);
    auto st = cat.Forward(inputs, outputs);
    ASSERT_EQ(st, InferStatus::kInferFailedDimensionParameterError);
}

TEST(test_cat_layer, InputSizeEqualOutputSizeError) {
    // inputs.size() == outputs.size，触发报错分支
    auto a = std::make_shared<Tensor<float>>(1,1,1);
    std::vector<std::shared_ptr<Tensor<float>>> inputs = {a};
    std::vector<std::shared_ptr<Tensor<float>>> outputs = {nullptr};

    CatLayer cat(1);
    auto st = cat.Forward(inputs, outputs);
    ASSERT_EQ(st, InferStatus::kInferFailedInputOutSizeMatchError);
}