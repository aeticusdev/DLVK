#pragma once

#include "layer.h"
#include "../tensor/tensor.h"
#include <memory>

namespace dlvk {

class DenseLayer : public Layer {
private:
    std::shared_ptr<Tensor> weights_;
    std::shared_ptr<Tensor> bias_;
    std::shared_ptr<Tensor> last_input_;
    std::shared_ptr<Tensor> grad_weights_;
    std::shared_ptr<Tensor> grad_bias_;
    size_t input_size_;
    size_t output_size_;
    VulkanDevice& device_;

public:
    DenseLayer(VulkanDevice& device, size_t input_size, size_t output_size);
    
    void initialize_weights();
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    void update_weights(float learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    // Getters for weights and bias (useful for training)
    std::shared_ptr<Tensor> get_weights() const { return weights_; }
    std::shared_ptr<Tensor> get_bias() const { return bias_; }
};

} // namespace dlvk
