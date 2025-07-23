#pragma once

#include "dlvk/layers/modern_layer.h"
#include "dlvk/tensor/tensor.h"

namespace dlvk {

/**
 * @brief Enumeration of activation function types
 */
enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax
};

/**
 * @brief Activation function layer
 */
class ActivationLayer : public ModernLayer {
private:
    ActivationType m_activation_type;
    std::shared_ptr<VulkanDevice> m_device;
    bool m_is_training;
    
public:
    /**
     * @brief Constructor for activation layer
     * @param device Vulkan device for tensor operations
     * @param activation_type Type of activation function
     */
    ActivationLayer(std::shared_ptr<VulkanDevice> device, ActivationType activation_type);
    
    /**
     * @brief Destructor
     */
    ~ActivationLayer() override = default;
    
    // Override virtual methods from Layer
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_parameters(Optimizer& optimizer) override;
    void set_training(bool training) override;
    LayerInfo get_layer_info() const override;
    void save_weights(std::ofstream& file) const override;
    void load_weights(std::ifstream& file) override;
    
    /**
     * @brief Get the activation type
     * @return The activation type
     */
    ActivationType get_activation_type() const { return m_activation_type; }
};

} // namespace dlvk
