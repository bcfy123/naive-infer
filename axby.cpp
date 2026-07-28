//
// Created by wjldw on 2026/7/29.
//

#include "ops.h"

void Axby(const arma::fmat &x, const arma::fmat &w, const arma::fmat &b,
          arma::fmat &y) {
    y = w*x + b;
}

void EPowerMinus(const arma::fmat &x, arma::fmat &y) {
    y = exp(-x);
}

void Axpy(const arma::fmat &x, arma::fmat &Y, float a, float y) {
    Y = a*x+y;
}
