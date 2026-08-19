//
// Created by twqb.
//

#include "runtime/runtime_attr.hpp"
namespace naive_infer {
void RuntimeAttribute::ClearWeight() {
  if (!this->weight_data.empty()) {
    std::vector<char> tmp = std::vector<char>();
    this->weight_data.swap(tmp);
  }
}
}  // namespace naive_infer