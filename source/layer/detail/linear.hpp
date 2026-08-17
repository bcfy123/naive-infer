//
// Created by twqb on 8/16/26.
//

#ifndef NAIVE_INFER_RELU_HPP
#define NAIVE_INFER_RELU_HPP
#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace naive_infer {
  class LinearLayer : public ParamLayer {
  public:
    explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                std::shared_ptr<Layer> &linear_layer);
  private:
    int32_t in_features_ = 0;
    int32_t out_features_ = 0;
    bool use_bias_ = false;
  };
}

#endif //NAIVE_INFER_RELU_HPP
