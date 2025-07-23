#pragma once

#include <memory>
#include <fstream>
#include <string>
#include "dlvk/tensor/tensor.h"

namespace dlvk {

class Optimizer;

/**
 * @brief Information about a layer for model summary
 */
struct LayerInfo {
    std::string type;
    std::string output_shape_str;
    size_t parameter_count;
    bool trainable;
    
    LayerInfo() : parameter_count(0), trainable(false) {}
};

/**
 * @brief Modern layer interface for high-level APIs
 */
class ModernLayer {
public:
    virtual ~ModernLayer() = default;
    
    /**
     * @brief Forward pass through the layer
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * @brief Backward pass through the layer
     * @param grad_output Gradient of the loss with respect to output
     * @return Gradient with respect to input
     */
    virtual Tensor backward(const Tensor& grad_output) = 0;
    
    /**
     * @brief Update layer parameters using the provided optimizer
     * @param optimizer Optimizer to use for parameter updates
     */
    virtual void update_parameters(Optimizer& optimizer) = 0;
    
    /**
     * @brief Set the layer to training or inference mode
     * @param training Whether the layer should be in training mode
     */
    virtual void set_training(bool training) = 0;
    
    /**
     * @brief Get information about this layer
     * @return LayerInfo structure with layer details
     */
    virtual LayerInfo get_layer_info() const = 0;
    
    /**
     * @brief Save layer weights to a file stream
     * @param file Output file stream
     */
    virtual void save_weights(std::ofstream& file) const = 0;
    
    /**
     * @brief Load layer weights from a file stream
     * @param file Input file stream
     */
    virtual void load_weights(std::ifstream& file) = 0;
};

} // namespace dlvk
