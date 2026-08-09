//
// Created by twqb on 8/10/26.
//

#ifndef NAIVEINFER_DATA_TENSOR_HPP
#define NAIVEINFER_DATA_TENSOR_HPP
#include <armadillo>
#include <vector>

namespace naive_infer {
template<typename T = float>
class Tensor {};

template<>
class Tensor<float> {
  public:
    explicit Tensor() = default;

    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    explicit Tensor(uint32_t size);

    explicit Tensor(uint32_t rows, uint32_t cols);

    explicit Tensor(const std::vector<uint32_t> &shapes);

    Tensor(const Tensor &tensor);

    Tensor(Tensor &&tensor) noexcept;

    Tensor<float> &operator=(Tensor &&tensor) noexcept;

    Tensor<float> &operator=(const Tensor<float> &tensor);

    uint32_t rows() const;

    uint32_t cols() const;

    uint32_t channels() const;

    uint32_t size() const;

    void set_data(const arma::fcube &data);

    bool empty() const;

    float index(uint32_t offset) const;

    float &index(uint32_t offset);

    std::vector<uint32_t> shapes() const;

    const std::vector<uint32_t> &raw_shapes() const;

    arma::fcube &data();

    const arma::fcube &data() const;

    arma::fmat &slice(uint32_t channel);

    const arma::fmat &slice(uint32_t channel) const;

    float at(uint32_t channel, uint32_t row, uint32_t col) const;

    float &at(uint32_t channel, uint32_t row, uint32_t col);

    void Padding(const std::vector<uint32_t> &pads, float padding_value);

    void Fill(float value);

    void Fill(const std::vector<float> &values, bool row_major = true);

    std::vector<float> values(bool row_major = true);

    void Ones();

    void Rand();

    void Show();

    void Reshape(const std::vector<uint32_t> &shapes, bool row_major=false);

    void Flatten(bool row_major=false);

    /**
     *
     * @param filter 一个可调用对象，接收 1 个 float 入参，返回 1 个 float
     */
    void Transform(const std::function<float(float)> &filter);

    float *raw_ptr();

    float *raw_ptr(uint32_t offset);

    float *matrix_raw_ptr(uint32_t index);

  private:
    std::vector<uint32_t> raw_shapes_;  // 张量数据的实际尺寸大小
    arma::fcube data_;                  // 张量数据
};
}

#endif //NAIVEINFER_DATA_TENSOR_HPP