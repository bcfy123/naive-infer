//
// Created by wjldw on 2026/7/29.
//

#ifndef OPS_H
#define OPS_H
#include <armadillo>

void Axby(const arma::fmat &x, const arma::fmat &w, const arma::fmat &b,
          arma::fmat &y);

void EPowerMinus(const arma::fmat &x, arma::fmat &y);

void Axpy(const arma::fmat &x, arma::fmat &Y, float a, float y);

#endif
