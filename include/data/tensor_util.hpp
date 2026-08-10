//
// Created by twqb on 8/10/26.
//

#ifndef NAIVEINFER_DATA_TENSOR_UTIL_HPP
#define NAIVEINFER_DATA_TENSOR_UTIL_HPP
#include "tensor.hpp"

namespace naive_infer {
  std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1, const sftensor& tensor2);

  std::shared_ptr<Tensor<float>> TensorPadding(const std::shared_ptr<Tensor<float>>& tensor,
    const std::vector<uint32_t>& pads, float padding_value);

  bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b,
                  float threshold = 1e-5f);

  std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

  void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor);

  void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                           const std::shared_ptr<Tensor<float>>& tensor2,
                           const std::shared_ptr<Tensor<float>>& output_tensor);

  std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

  std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols);

  std::shared_ptr<Tensor<float>> TensorCreate(
    const std::vector<uint32_t>& shapes);

  std::shared_ptr<Tensor<float>> TensorClone(
    std::shared_ptr<Tensor<float>> tensor);
}

#endif //NAIVEINFER_DATA_TENSOR_UTIL_HPP