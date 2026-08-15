//
// Created by twqb on 8/15/26.
//

#ifndef NAIVEINFER_TENSOR_HPP
#define NAIVEINFER_TENSOR_HPP
#include <cstdint>

namespace naive_infer {
  template <typename T = float>
  class Tensor {};

  template<>
  class Tensor<float> {
  public:
    explicit Tensor() = default;

    /**
     * create a 3-dimensional tensor
     * @param channels
     * @param rows
     * @param cols
     */
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    /**
     * create a 2-dimensional tensor
     * @param rows
     * @param cols
     */
    explicit Tensor(uint32_t rows, uint32_t cols);

    /**
     * create a 1-dimensional tensor
     * @param size
     */
    explicit Tensor(uint32_t size);

  private:
    std::vector<uint32_t> raw_shapes_;

    arma::fcube data_;
  };
}

#endif //NAIVEINFER_TENSOR_HPP