//
// Created by twqb on 8/16/26.
//
#include "layer/abstract/layer.hpp"

namespace naive_infer {
  const std::vector<std::shared_ptr<Tensor<float>>>& Layer::weights() const {
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  }

  const std::vector<std::shared_ptr<Tensor<float>>>& Layer::bias() const {
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  }

  void Layer::set_bias(const std::vector<float>& bias) {
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  }

  void Layer::set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) {
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  }

  void Layer::set_weights(const std::vector<float>& weights) {
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  }

  void Layer::set_weights(
      const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  }

  InferStatus Layer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  }

  InferStatus Layer::Forward() {
    LOG_IF(FATAL, this->runtime_operator_.expired())
        << "Runtime operator is expired or nullptr";
    const auto& runtime_operator = this->runtime_operator_.lock();
    // Prepare inputs required for layer computation
    const std::vector<std::shared_ptr<RuntimeOperand>>& input_operand_datas =
        runtime_operator->input_operands_seq;
    // Inputs of the layer
    std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;
    for (const auto& input_operand_data : input_operand_datas) {
      for (const auto& input_data : input_operand_data->datas) {
        layer_input_datas.push_back(input_data);
      }
    }
    // outputs of the layer
    const std::shared_ptr<RuntimeOperand>& output_operand_datas =
      runtime_operator->output_operands;

    CHECK(!layer_input_datas.empty())
        << runtime_operator->name << " Layer input data is empty";
    CHECK(output_operand_datas != nullptr && !output_operand_datas->datas.empty())
        << "Layer output data is empty";
    // Execute the forward computation of the layer within the operator
    // Layer computation results are stored in current_op->output_operands->datas
    InferStatus status = runtime_operator->layer->Forward(
        layer_input_datas, output_operand_datas->datas);
    return status;
  }

  void Layer::set_runtime_operator(
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
    CHECK(runtime_operator != nullptr);
    this->runtime_operator_ = runtime_operator;
  }
}