//
// Created by twqb on 8/2/26.
//

#include "tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

float MinusOne(float value) { return value - 1.f; }
TEST(test_transform, transform1) {
    using namespace naive_infer;
    Tensor<float> f1(2, 3, 4);
    f1.Rand();
    f1.Show();
    f1.Transform(MinusOne);
    f1.Show();
}