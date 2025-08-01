#pragma once

#include <memory>
#include "dlvk/tensor/tensor.h"

namespace dlvk {

class VulkanDevice;

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::unique_ptr<Layer> clone() const = 0;
    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) = 0;
    virtual std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) = 0;
    virtual void update_weights(float learning_rate) {}
    
protected:
    std::shared_ptr<VulkanDevice> m_device;
};

class ConvLayer : public Layer {
public:
    ConvLayer(size_t in_channels, size_t out_channels, size_t kernel_size, 
              std::shared_ptr<VulkanDevice> device);
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
private:
    size_t m_in_channels;
    size_t m_out_channels;
    size_t m_kernel_size;
    
    std::shared_ptr<Tensor> m_weights;
    std::shared_ptr<Tensor> m_bias;
};

} // namespace dlvk
