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

    /**
     * get data of the given channel from the tensor
     * @param channel
     * @return
     */
    arma::fmat &slice(uint32_t channel);

    /**
     * get data of the given channel from the tensor
     * @param channel
     * @return
     */
    const arma::fmat &slice(uint32_t channel) const;

    /**
     *
     * @param channel
     * @param row
     * @param col
     * @return element at the given 3D coordinate
     */
    float at(uint32_t channel, uint32_t row, uint32_t col) const;

    /**
     *
     * @param channel
     * @param row
     * @param col
     * @return element at the given 3D coordinate
     */
    float &at(uint32_t channel, uint32_t row, uint32_t col);

    /**
     * get raw pointer to the underlying tensor data.
     * @return pointer to the start of the data buffer
     */
    float *raw_ptr();

    /**
     * get raw pointer at the specified offset of the underlying tensor data.
     * @param offset element offset from the data start
     * @return pointer to the data position at the given offset
     */
    float *raw_ptr(uint32_t offset);

    /**
     * get raw pointer of the matrix at the specified index.
     * @param index index of the target matrix
     * @return raw pointer to the selected matrix data
     */
    float *matrix_raw_ptr(uint32_t index);

    /**
     * Initialize tensor data with the value.
     * @param value used to fill the tensor
     */
    void Fill(float value);

    /**
     * Initialize tensor data from the given values.
     * @param values data used to initialize the tensor
     * @param row_major If true, interpret input values in row-major order
     */
    void Fill(const std::vector<float> &values, bool row_major = true);

  private:
    std::vector<uint32_t> raw_shapes_;

    arma::fcube data_;
  };
}

#endif //NAIVEINFER_TENSOR_HPP