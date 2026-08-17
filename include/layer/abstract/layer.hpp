//
// Created by twqb on 8/15/26.
//

#ifndef NAIVEINFER_LAYER_HPP
#define NAIVEINFER_LAYER_HPP
#include <glog/logging.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "data/tensor.hpp"
#include "runtime/runtime_op.hpp"
#include "status_code.hpp"

namespace naive_infer {
  class RuntimeOperator;
  class Layer {
    public:
      explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}

      virtual ~Layer() = default;

      /**
      * Performs layer forward inference
      * @param inputs Vector of input tensors
      * @param outputs Vector to store output tensors
      * @return Status code indicating whether forward inference succeeded
      */
      virtual InferStatus Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs);

      /**
       * Performs layer forward inference
       * @return Status code indicating whether forward inference succeeded
       */
      virtual InferStatus Forward();

      /**
       * get weights of the layer
       * @return
       */
      virtual const std::vector<std::shared_ptr<Tensor<float>>>& weights() const;

      /**
      * get biases of the layer
      * @return
      */
      virtual const std::vector<std::shared_ptr<Tensor<float>>>& bias() const;

      /**
       * set weights of the layer
       * @param weights
       */
      virtual void set_weights(
          const std::vector<std::shared_ptr<Tensor<float>>>& weights);

      /**
       * set biases of the layer
       * @param bias
       */
      virtual void set_bias(
          const std::vector<std::shared_ptr<Tensor<float>>>& bias);

      /**
       * set weights of the layer
       * @param weights
       */
      virtual void set_weights(const std::vector<float>& weights);

      /**
       * set biases of the layer
       * @param bias
       */
      virtual void set_bias(const std::vector<float>& bias);

      /**
       * get the name of the layer
       * @return
       */
      virtual const std::string& layer_name() const { return this->layer_name_; }

      /**
       * get the operator of the layer
       * @param runtime_operator
       */
      void set_runtime_operator(
          const std::shared_ptr<RuntimeOperator>& runtime_operator);

  protected:
    std::weak_ptr<RuntimeOperator> runtime_operator_;
    std::string layer_name_;  /// layer name
  };
}

#endif //NAIVEINFER_LAYER_HPP