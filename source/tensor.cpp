//
// Created by twqb on 8/16/26.
//
#include "data/tensor.hpp"
#include <glog/logging.h>

namespace naive_infer {
  Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    data_ = arma::fcube(channels, rows, cols);
    if (channels==1&&rows==1) {
      raw_shapes_ = std::vector<uint32_t>{cols};
    }else if (channels==1) {
      raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }else {
      raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
    }
  }

  Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
    data_ = arma::fcube(rows, cols, 1);
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  }

  Tensor<float>::Tensor(uint32_t size) {
    data_ = arma::fcube(size, 1, 1);
    this->raw_shapes_ = std::vector<uint32_t>{size};
  }

  Tensor<float>::Tensor(const std::vector<uint32_t> &shapes) {
    CHECK(!shapes.empty() && shapes.size()<=3);
    uint32_t remaining = 3-shapes.size();
    std::vector<uint32_t> shapes_(3,1);
    std::copy(shapes.begin(), shapes.end(), shapes_.begin()+remaining);

    uint32_t channels = shapes_.at(0);
    uint32_t rows = shapes_.at(1);
    uint32_t cols = shapes_.at(2);

    data_ = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows==1) {
      this->raw_shapes_ = std::vector<uint32_t>{cols};
    }else if (channels == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }else {
      this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
    }
  }

  Tensor<float>::Tensor(const Tensor<> &tensor) {
    if (this!=&tensor) {
      this->data_ = tensor.data_;
      this->raw_shapes_ = tensor.raw_shapes_;
    }
  }

  Tensor<float>::Tensor(Tensor<> &&tensor) noexcept {
    if (this!=&tensor) {
      this->data_ = std::move(tensor.data_);
      this->raw_shapes_ = std::move(tensor.raw_shapes_);
    }
  }

  Tensor<float> & Tensor<float>::operator=(Tensor<> &&tensor) noexcept {
    if (this!=&tensor) {
      this->data_ = std::move(tensor.data_);
      this->raw_shapes_ = std::move(tensor.raw_shapes_);
    }
  }

  Tensor<float> & Tensor<float>::operator=(const Tensor<> &tensor) {
    if (this!=&tensor) {
      this->data_ = tensor.data_;
      this->raw_shapes_ = tensor.raw_shapes_;
    }
  }

  uint32_t Tensor<float>::rows() const {
    CHECK(!this->data_.empty());
    return this->data_.n_rows;
  }

  uint32_t Tensor<float>::cols() const {
    CHECK(!this->data_.empty());
    return this->data_.n_cols;
  }

  uint32_t Tensor<float>::channels() const {
    CHECK(!this->data_.empty());
    return this->data_.n_slices;
  }

  uint32_t Tensor<float>::size() const {
    CHECK(!this->data_.empty());
    return this->data_.size();
  }

  std::vector<uint32_t> Tensor<float>::shapes() const {
    CHECK(!this->data_.empty());
    return {this->channels(), this->rows(), this->cols()};
  }

  const std::vector<uint32_t> & Tensor<float>::raw_shapes() const {
    CHECK(!this->raw_shapes_.empty());
    CHECK_LE(this->raw_shapes_.size(), 3);
    CHECK_GE(this->raw_shapes_.size(), 1);
    return this->raw_shapes_;
  }

  const arma::fcube & Tensor<float>::data() const {
    return this->data_;
  }

  void Tensor<float>::set_data(const arma::fcube &data) {
    CHECK(data.n_rows == this->data_.n_rows)
        << data.n_rows << " != " << this->data_.n_rows;
    CHECK(data.n_cols == this->data_.n_cols)
            << data.n_cols << " != " << this->data_.n_cols;
    CHECK(data.n_slices == this->data_.n_slices)
            << data.n_slices << " != " << this->data_.n_slices;
    this->data_ = data;
  }

  bool Tensor<float>::empty() const {
    return this->data_.empty();
  }

  float Tensor<float>::index(uint32_t offset) const {
    CHECK(offset < this->data_.size());
    return this->data_.at(offset);
  }

  float & Tensor<float>::index(uint32_t offset) {
    CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
    return this->data_.at(offset);
  }

  arma::fcube & Tensor<float>::data() {
    return this->data_;
  }
}
