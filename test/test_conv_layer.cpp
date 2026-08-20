//
// Created by twqb on 8/20/26.
//

#include <gtest/gtest.h>
#include "../source/layer/detail/convolution.hpp"
#include "data/tensor_util.hpp"
#include <vector>

using namespace naive_infer;

TEST(TestConvolutionLayer, Conv2dBasicWithBias) {
    // 1. 构造卷积层: out=2, in=2, kernel 3x3, pad0, stride1, groups=1, use_bias=true
    uint32_t out_c = 2;
    uint32_t in_c  = 2;
    uint32_t k_h = 3, k_w =3;
    uint32_t pad_h=0, pad_w=0;
    uint32_t s_h=1, s_w=1;
    uint32_t groups =1;
    bool use_bias = true;

    auto conv = std::make_shared<ConvolutionLayer>(
        out_c, in_c, k_h, k_w, pad_h, pad_w, s_h, s_w, groups, use_bias
    );

    // ---------------------- 2. 设置权重 weight [out_c,in_c,3,3] ----------------------
    // weight0: out_channel 0，两个输入通道的3x3 kernel全部填1
    std::vector<float> weight_data(2*3*3 * out_c, 1.0f);
    conv->set_weights(weight_data);

    // bias: out_c=2，bias0=0.0f, bias1=1.0f
    std::vector<float> bias_data{0.0f, 1.0f};
    conv->set_bias(bias_data);

    // 预执行im2col权重预处理，和GetInstance里面逻辑对齐
    conv->InitIm2ColWeight();

    // ---------------------- 3.构造输入张量 batch=1, [1, 2, 4,4] ----------------------
    auto input = TensorCreate(2, 4, 4);
    // 通道0全部赋值1；通道1全部赋值1
    for(uint32_t c=0;c<2;c++){
        for(uint32_t r=0;r<4;r++){
            for(uint32_t col=0;col<4;col++){
                input->at(c,r,col) = 1.0f;
            }
        }
    }

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    // outputs预先放入nullptr，让Forward内部创建输出张量
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    outputs.emplace_back(nullptr);

    // ---------------------- 4.执行前向 ----------------------
    InferStatus status = conv->Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_NE(outputs.at(0), nullptr);

    // ---------------------- 5.校验输出shape ----------------------
    auto out_tensor = outputs.at(0);
    // N=1, C=2, H=2, W=2
    ASSERT_EQ(out_tensor->channels(), 2);
    ASSERT_EQ(out_tensor->rows(), 2);
    ASSERT_EQ(out_tensor->cols(), 2);
    LOG(INFO) << "output data: " << out_tensor->data();
}