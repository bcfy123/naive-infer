//
// Created by fss on 23-8-5.
//
#include <gtest/gtest.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../source/layer/details/expression.hpp"
#include "runtime/runtime_ir.hpp"
#include "../source/layer/details/softmax.hpp"

// 引入推理框架命名空间，简化类型书写
using namespace naive_infer;

/**
 * @brief 图像预处理函数，将OpenCV原图转换成ResNet18模型标准输入张量
 * @param image OpenCV读取的原始BGR格式图像
 * @return sftensor 推理输入张量，维度排布 [C, H, W] = [3, 224, 224]
 * @detail 严格对齐ImageNet预训练模型训练时的预处理流程
 */
naive_infer::sftensor PreProcessImage(const cv::Mat &image) {
    // 函数内再次引入推理命名空间，避免类型全限定书写
    using namespace naive_infer;
    // 合法性校验：输入图片不能为空
    assert(!image.empty());

    // 1. 将原图缩放至模型固定输入尺寸 224*224
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(224, 224));

    // 2. OpenCV默认读取顺序BGR，转换成深度学习通用RGB通道顺序
    cv::Mat rgb_image;
    cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

    // 3. 图像uint8像素转为32位浮点数，支持浮点归一化运算
    rgb_image.convertTo(rgb_image, CV_32FC3);
    // 将RGB三通道图像拆分为三张单通道Mat
    std::vector<cv::Mat> split_images;
    cv::split(rgb_image, split_images);

    // 模型固定输入维度：通道数3，高224，宽224
    uint32_t input_w = 224;
    uint32_t input_h = 224;
    uint32_t input_c = 3;
    // 创建推理框架张量，格式 NCHW 单样本即 [C, H, W]
    sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

    uint32_t index = 0;
    // 遍历RGB三个通道，将图像数据拷贝到张量对应通道内存
    for (const auto &split_image : split_images) {
        // 单通道图像像素总数必须等于224*224
        assert(split_image.total() == input_w * input_h);
        // 矩阵转置，匹配框架张量的内存排布规则
        const cv::Mat &split_image_t = split_image.t();
        // 内存拷贝：将OpenCV通道数据直接复制到张量通道内存
        memcpy(input->slice(index).memptr(), split_image_t.data,
               sizeof(float) * split_image.total());
        index += 1;
    }

    // ImageNet数据集 RGB 通道均值
    float mean_r = 0.485f;
    float mean_g = 0.456f;
    float mean_b = 0.406f;
    // ImageNet数据集 RGB 通道标准差
    float var_r = 0.229f;
    float var_g = 0.224f;
    float var_b = 0.225f;

    // 校验张量通道数为3（RGB图像）
    assert(input->channels() == 3);
    // 像素值从0~255归一化至0~1区间
    input->data() = input->data() / 255.f;
    // 逐通道标准化 (x - mean) / std，和训练预处理保持一致
    input->slice(0) = (input->slice(0) - mean_r) / var_r; // R通道标准化
    input->slice(1) = (input->slice(1) - mean_g) / var_g; // G通道标准化
    input->slice(2) = (input->slice(2) - mean_b) / var_b; // B通道标准化

    return input;
}

/**
 * @brief GTest单元测试用例：ResNet18图像分类完整推理流程
 * 测试链路：加载PNNX模型 → 图像预处理 → 前向推理 → Softmax概率归一化 → 输出最优分类
 */
TEST(test_network, resnet1) {
    using namespace naive_infer;
    // PNNX模型文件路径：param存储网络结构，bin存储模型权重
    const std::string &param_path = "../../model_file/resnet18_batch1.pnnx.param";
    const std::string &weight_path = "../../model_file/resnet18_batch1.pnnx.bin";

    // 创建推理计算图对象，加载模型结构与权重
    RuntimeGraph graph(param_path, weight_path);
    // 构建完整推理IR图，指定PNNX导出的输入节点、输出节点名称
    graph.Build("pnnx_input_0", "pnnx_output_0");

    // 推理批次大小，当前仅单张图片推理
    const uint32_t batch_size = 1;
    // 存储批量推理输入张量
    std::vector<sftensor> inputs;
    // 待推理测试图片路径
    const std::string &path("../../course8/model_file/car.jpg");

    // 循环构造batch输入（当前batch=1，仅加载一张图片）
    for (uint32_t i = 0; i < batch_size; ++i) {
        // 读取本地测试图片
        cv::Mat image = cv::imread(path);
        // 调用预处理函数，生成模型可接受的张量
        sftensor input = PreProcessImage(image);
        inputs.push_back(input);
    }

    // 执行模型前向推理，第二个参数true开启CPU多线程加速
    auto outputs = graph.Forward(inputs, true);
    // GTest断言：输出batch数量必须和输入batch数量相等
    ASSERT_EQ(outputs.size(), batch_size);

    // 实例化Softmax层，沿第0维（类别维度）做归一化
    SoftmaxLayer softmax_layer(0);
    // 容器存放Softmax归一化后的概率张量
    std::vector<sftensor> outputs_softmax(batch_size);
    // 对模型原始输出logits做Softmax，转换成0~1概率分布
    softmax_layer.Forward(outputs, outputs_softmax);
    assert(outputs_softmax.size() == batch_size);

    // 遍历每个batch样本，解析分类预测结果
    for (int i = 0; i < outputs_softmax.size(); ++i) {
        const sftensor &output_tensor = outputs_softmax.at(i);
        // ResNet18在ImageNet数据集输出固定1000个分类类别
        assert(output_tensor->size() == 1 * 1000);

        float max_prob = -1; // 记录当前最大类别概率
        int max_index = -1;  // 最大概率对应的类别下标

        // 遍历全部1000个类别，查找概率最大值
        for (int j = 0; j < output_tensor->size(); ++j) {
            float prob = output_tensor->index(j);
            if (max_prob <= prob) {
                max_prob = prob;
                max_index = j;
            }
        }
        // 打印预测最高概率、对应类别索引
        printf("class with max prob is %f index %d\n", max_prob, max_index);
    }
}