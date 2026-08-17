//
// Created by twqb on 8/16/26.
//
#include <map>
#include <memory>
#include <string>
#include "layer.hpp"
#include "runtime/runtime_op.hpp"

#ifndef NAIVE_INFER_LAYER_FACTORY_HPP
#define NAIVE_INFER_LAYER_FACTORY_HPP

namespace naive_infer {
  class LayerRegisterer {
  public:
    typedef ParseParameterAttrStatus (*Creator)(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &layer);

    typedef std::map<std::string, Creator> CreateRegistry;

  /**
   * Registers an operator creator into the registry
   * @param layer_type Type identifier of the operator
   * @param creator Creator function for the target operator
   */
  static void RegisterCreator(const std::string &layer_type,
                              const Creator &creator);

  /**
   * Initializes a layer using information from the runtime operator
   * @param op Runtime operator containing layer initialization parameters
   * @return Constructed layer instance
   */
  static std::shared_ptr<Layer> CreateLayer(
      const std::shared_ptr<RuntimeOperator> &op);

  /**
   * Gets the operator creator registry
   * @return Reference to the creator registry
   */
  static CreateRegistry &Registry();

  /**
   * Obtains the type list of all registered operators
   * @return List of registered layer types
   */
  static std::vector<std::string> layer_types();
  };

  class LayerRegistererWrapper {
  public:
    LayerRegistererWrapper(const std::string &layer_type,
                         const LayerRegisterer::Creator &creator) {
      LayerRegisterer::RegisterCreator(layer_type, creator);
    }
  };
}

#endif //NAIVE_INFER_LAYER_FACTORY_HPP