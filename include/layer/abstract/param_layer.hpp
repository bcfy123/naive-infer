//
// Created by twqb on 8/16/26.
//

#ifndef NAIVE_INFER_PARAM_LAYER_HPP
#define NAIVE_INFER_PARAM_LAYER_HPP
#include "layer.hpp"

namespace naive_infer {
  class ParamLayer : public Layer {
  public:
    explicit ParamLayer(const std::string &layer_name);
    /**
     * initialize weight parameters
     * @param param_count   Number
     * @param param_channel  input channels
     * @param param_height   rows
     * @param param_width    cols
     */
    void InitWeightParam(const uint32_t param_count, const uint32_t param_channel,
                         const uint32_t param_height, const uint32_t param_width);

    /**
     * initialize bias parameters
     * @param param_count    Number of offset parameters
     * @param param_channel  Channel count of offset parameters
     * @param param_height   Height of offset parameters
     * @param param_width    Width of offset parameters
     */
    void InitBiasParam(const uint32_t param_count, const uint32_t param_channel,
                       const uint32_t param_height, const uint32_t param_width);

    /**
     * get weights
     * @return
     */
    const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

    /**
     * get biases
     * @return
     */
    const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

    /**
     * set weights
     * @param weights
     */
    void set_weights(const std::vector<float> &weights) override;

    /**
     * set biases
     * @param bias 偏移量参数
     */
    void set_bias(const std::vector<float> &bias) override;

    /**
     * set weights
     * @param weights
     */
    void set_weights(
        const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

    /**
     * set biases
     * @param bias
     */
    void set_bias(
        const std::vector<std::shared_ptr<Tensor<float>>> &bias) override;

  protected:
    std::vector<std::shared_ptr<Tensor<float>>> weights_;
    std::vector<std::shared_ptr<Tensor<float>>> bias_;
  };
}

#endif //NAIVE_INFER_PARAM_LAYER_HPP
