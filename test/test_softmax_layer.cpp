//
// Created by twqb on 8/22/26.
//

#include <gtest/gtest.h>
#include "data/tensor.hpp"
#include "../source/layer/detail/softmax.hpp"
#include <vector>
#include <cmath>

using namespace naive_infer;

// 适配 arma::fcube 的张量比对工具函数
bool TensorEqual(const sftensor& a, const sftensor& b, float eps = 1e-4f) {
  if (a->shapes() != b->shapes()) {
    return false;
  }
  const arma::fcube& cube_a = a->data();
  const arma::fcube& cube_b = b->data();
  if (cube_a.n_elem != cube_b.n_elem) {
    return false;
  }

  const float* ptr_a = cube_a.memptr();
  const float* ptr_b = cube_b.memptr();
  for (arma::uword i = 0; i < cube_a.n_elem; ++i) {
    if (std::fabs(ptr_a[i] - ptr_b[i]) > eps) {
      return false;
    }
  }
  return true;
}

TEST(TestSoftmaxLayer, SoftmaxDim1) {
  std::vector<sftensor> inputs;
  sftensor in_tensor = std::make_shared<Tensor<float>>(std::vector<uint32_t>{2, 3});
  std::vector<float> in_data = {1.f, 2.f, 3.f,
                                4.f, 5.f, 6.f};
  in_tensor->Fill(in_data);
  inputs.push_back(in_tensor);

  std::vector<sftensor> outputs;
  SoftmaxLayer softmax_layer(1);
  softmax_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  const sftensor& out = outputs.front();

  std::vector<float> ref_data = {
    0.09003058f, 0.24472847f, 0.66524094f,
    0.09003058f, 0.24472847f, 0.66524094f
  };
  sftensor ref_tensor = std::make_shared<Tensor<float>>(std::vector<uint32_t>{2, 3});
  ref_tensor->Fill(ref_data);

  ASSERT_TRUE(TensorEqual(out, ref_tensor));
}
