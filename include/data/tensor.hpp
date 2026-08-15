//
// Created by twqb on 8/15/26.
//

#ifndef NAIVEINFER_TENSOR_HPP
#define NAIVEINFER_TENSOR_HPP
#include <armadillo>

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

    /**
     * create a tensor from the given shape dimensions.
     * @param shapes
     */
    explicit Tensor(const std::vector<uint32_t> &shapes);

    Tensor(const Tensor &tensor);

    Tensor(Tensor &&tensor) noexcept;

    Tensor<float> &operator=(Tensor &&tensor) noexcept;

    Tensor<float> &operator=(const Tensor &tensor);

    /**
     * @return the number of rows
     */
    uint32_t rows() const;

    /**
     * @return the number of columns
     */
    uint32_t cols() const;

    /**
     * @return the number of channels
     */
    uint32_t channels() const;

    /**
     * @return the number of elements
     */
    uint32_t size() const;

    /**
     * @return the shape of the tensor
     */
    std::vector<uint32_t> shapes() const;

    /**
     * @return the raw shape of the tensor
     */
    const std::vector<uint32_t> &raw_shapes() const;

    /**
     * @return the underlying tensor data
     */
    arma::fcube &data();

    /**
     * 返回张量中的数据
     * @return 张量中的数据
     */
    const arma::fcube &data() const;

    /**
     * set the underlying tensor data
     * @param data
     */
    void set_data(const arma::fcube &data);

    /**
     *
     * @return check if the tensor is empty
     */
    bool empty() const;

    /**
     * get the element at the given index
     * @param offset index position to access
     * @return element of the given index
     */
    float index(uint32_t offset) const;

    /**
     * get the element at the given index
     * @param offset index position to access
     * @return element of the given index
     */
    float &index(uint32_t offset);

  private:
    std::vector<uint32_t> raw_shapes_;

    arma::fcube data_;
  };
}

#endif //NAIVEINFER_TENSOR_HPP