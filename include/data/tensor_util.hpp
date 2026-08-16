//
// Created by twqb on 8/16/26.
//

#ifndef NAIVE_INFER_TENSOR_UTIL_HPP
#define NAIVE_INFER_TENSOR_UTIL_HPP
#include "data/tensor.hpp"

namespace naive_infer {
  /**
   * broadcast two tensors to compatible shapes.
   * @param tensor1 first input tensor
   * @param tensor2 second input tensors
   * @return tuple of two tensors with broadcast-compatible shapes
   */
  std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1,
                                                 const sftensor& tensor2);

  /**
   * pads a tensor.
   * @param tensor Input tensor to be padded
   * @param pads Padding sizes
   * @param padding_value Value used for padding
   * @return Tensor after padding
   */
  std::shared_ptr<Tensor<float>> TensorPadding(
      const std::shared_ptr<Tensor<float>>& tensor,
      const std::vector<uint32_t>& pads, float padding_value);

  /**
 * Checks whether two tensors have identical values within a threshold.
 * @param a First input tensor
 * @param b Second input tensor
 * @param threshold Threshold for element-wise value difference
 * @return True if tensors are considered identical, otherwise false
 */
bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b,
                  float threshold = 1e-5f);

/**
 * Performs element-wise addition of two tensors.
 * @param tensor1 First input tensor
 * @param tensor2 Second input tensor
 * @return Result tensor of element-wise addition
 */
std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * Performs element-wise addition of two tensors and writes result to output tensor.
 * @param tensor1 First input tensor
 * @param tensor2 Second input tensor
 * @param output_tensor Output tensor to store the addition result
 */
void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * Performs element-wise multiplication of two tensors and writes result to output tensor.
 * @param tensor1 First input tensor
 * @param tensor2 Second input tensor
 * @param output_tensor Output tensor to store the multiplication result
 */
void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                           const std::shared_ptr<Tensor<float>>& tensor2,
                           const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * Performs element-wise multiplication of two tensors.
 * @param tensor1 First input tensor
 * @param tensor2 Second input tensor
 * @return Result tensor of element-wise multiplication
 */
std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * Creates a 3D tensor.
 * @param channels Number of channels
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Created tensor
 */
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols);

/**
 * Creates a tensor with the specified shape.
 * @param shapes Shape of the tensor
 * @return Created tensor
 */
std::shared_ptr<Tensor<float>> TensorCreate(
    const std::vector<uint32_t>& shapes);

/**
 * Returns a deep copy of the input tensor.
 * @param tensor Tensor to clone
 * @return New deep-copied tensor
 */
std::shared_ptr<Tensor<float>> TensorClone(
    std::shared_ptr<Tensor<float>> tensor);

}

#endif //NAIVE_INFER_TENSOR_UTIL_HPP