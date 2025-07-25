#pragma once

#include "dlvk/layers/modern_layer.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/core/vulkan_device.h"

namespace dlvk {

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax
};

class ActivationLayer : public ModernLayer {
private:
    ActivationType m_activation_type;
    std::shared_ptr<VulkanDevice> m_device;
    bool m_is_training;
    
public:
    ActivationLayer(std::shared_ptr<VulkanDevice> device, ActivationType activation_type);
    
    std::unique_ptr<ModernLayer> clone() const override;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_parameters([[maybe_unused]] Optimizer& optimizer) override;
    void set_training(bool training) override;
    LayerInfo get_layer_info() const override;
    void save_weights([[maybe_unused]] std::ofstream& file) const override;
    void load_weights([[maybe_unused]] std::ifstream& file) override;
    
    ActivationType get_activation_type() const { return m_activation_type; }
};

} // namespace dlvk