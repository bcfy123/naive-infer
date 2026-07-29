//
// Created by wjldw on 2026/7/29.
//

#include "ops.hpp"
#include <armadillo>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(test_arma, Axby) {
  using namespace arma;
  fmat w = "1,2,3;"
           "4,5,6;"
           "7,8,9;";

  fmat x = "1,2,3;"
           "4,5,6;"
           "7,8,9;";

  fmat b = "1,1,1;"
           "2,2,2;"
           "3,3,3;";

  fmat answer = "31,37,43;"
                "68,83,98;"
                "105,129,153";

  fmat y;
  Axby(x, w, b, y);
  ASSERT_EQ(approx_equal(y, answer, "absdiff", 1e-5f), true);
}

TEST(test_arma, e_power_minus) {
  using namespace arma;
  fmat x(224, 224);
  x.randu();

  fmat y;
  EPowerMinus(x, y);
  ASSERT_FALSE(y.empty());

  const uword elem_num = x.n_elem;
  for (uword i = 0; i < elem_num; ++i) {
    float expect = std::exp(-x.at(i));
    ASSERT_LE(std::abs(expect - y.at(i)), 1e-5f);
  }
}

TEST(test_arma, axpy) {
  using namespace arma;
  fmat x(224, 224);
  x.randu();

  fmat Y;
  float a = 3.f;
  float y = 4.f;
  Axpy(x, Y, a, y);

  ASSERT_FALSE(Y.empty());
  const uword elem_num = x.n_elem;
  for (uword i = 0; i < elem_num; ++i) {
    float expect = a * x.at(i) + y;
    ASSERT_LE(std::abs(expect - Y.at(i)), 1e-5f);
  }
}

TEST(test_arma, add) {
  using namespace arma;
  fmat in_matrix1 = "1,2,3;"
                    "4,5,6;"
                    "7,8,9";

  fmat in_matrix2 = "1,2,3;"
                    "4,5,6;"
                    "7,8,9";

  const fmat out_matrix1 = "2,4,6;"
                            "8,10,12;"
                            "14,16,18";

  const fmat out_matrix2 = in_matrix1 + in_matrix2;
  ASSERT_EQ(approx_equal(out_matrix1, out_matrix2, "absdiff", 1e-5), true);
}

TEST(test_arma, sub) {
  using namespace arma;
  fmat in_matrix1 = "1,2,3;"
                    "4,5,6;"
                    "7,8,9";

  fmat in_matrix2 = "1,2,3;"
                    "4,5,6;"
                    "7,8,9";

  const fmat out_matrix1 = "0,0,0;"
                            "0,0,0;"
                            "0,0,0;";

  const fmat out_matrix2 = in_matrix1 - in_matrix2;
  ASSERT_EQ(approx_equal(out_matrix1, out_matrix2, "absdiff", 1e-5), true);
}

TEST(test_arma, matmul) {
  using namespace arma;
  fmat in_matrix1 = "1,2,3;"
                    "4,5,6;"
                    "7,8,9";

  fmat in_matrix2 = "1,2,3;"
                    "4,5,6;"
                    "7,8,9";

  const fmat out_matrix1 = "30,36,42;"
                            "66,81,96;"
                            "102,126,150;";

  const fmat out_matrix2 = in_matrix1 * in_matrix2;
  ASSERT_EQ(approx_equal(out_matrix1, out_matrix2, "absdiff", 1e-5), true);
}

TEST(test_arma, pointwise) {
  using namespace arma;
  fmat in_matrix1 = "1,2,3;"
                    "4,5,6;"
                    "7,8,9";

  fmat in_matrix2 = "1,2,3;"
                    "4,5,6;"
                    "7,8,9";

  const fmat out_matrix1 = "1,4,9;"
                            "16,25,36;"
                            "49,64,81;";

  const fmat out_matrix2 = in_matrix1 % in_matrix2;
  ASSERT_EQ(approx_equal(out_matrix1, out_matrix2, "absdiff", 1e-5), true);
}