//
// Created by twqb on 8/5/26.
//
#include "tensor.hpp"
#include "include/runtime/ir.h"
// #include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>
static std::string ShapeStr(const std::vector<int> &shapes) {
    std::ostringstream ss;
    for (int i = 0; i < shapes.size(); ++i) {
        ss << shapes.at(i);
        if (i != shapes.size() - 1) {
            ss << " x ";
        }
    }
    return ss.str();
}

TEST(test_ir, pnnx_graph_ops) {
    using namespace naive_infer;
    /**
     * 如果这里加载失败，请首先考虑相对路径的正确性问题
     */
    std::string bin_path("course3/model_file/test_linear.pnnx.bin");
    std::string param_path("course3/model_file/test_linear.pnnx.param");
    std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
    int load_result = graph->load(param_path, bin_path);
    // 如果这里加载失败，请首先考虑相对路径(bin_path和param_path)的正确性问题
    ASSERT_EQ(load_result, 0);
    const auto &ops = graph->ops;
    for (int i = 0; i < ops.size(); ++i) {
        LOG(INFO) << ops.at(i)->name;
    }
}