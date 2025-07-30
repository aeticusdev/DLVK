#pragma once

#include "layer.h"
#include "../tensor/tensor.h"
#include <memory>

namespace dlvk {

class Conv2DLayer : public Layer {
private:
    std::shared_ptr<Tensor> weights_;        // Shape: [out_channels, in_channels, kernel_h, kernel_w]
    std::shared_ptr<Tensor> bias_;           // Shape: [out_channels]
    std::shared_ptr<Tensor> weight_grads_;   // Gradients for weights
    std::shared_ptr<Tensor> bias_grads_;     // Gradients for bias
    std::shared_ptr<Tensor> last_input_;     // For backward pass
    
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_height_;
    size_t kernel_width_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;
    VulkanDevice& device_;

public:
    Conv2DLayer(VulkanDevice& device, 
                size_t in_channels, size_t out_channels,
                size_t kernel_height, size_t kernel_width,
                size_t stride_h = 1, size_t stride_w = 1,
                size_t padding_h = 0, size_t padding_w = 0);
    
    void initialize_weights();
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    void update_weights(float learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    

    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const;
    

    std::shared_ptr<Tensor> get_weights() const { return weights_; }
    std::shared_ptr<Tensor> get_bias() const { return bias_; }
    
    size_t get_in_channels() const { return in_channels_; }
    size_t get_out_channels() const { return out_channels_; }
    size_t get_kernel_height() const { return kernel_height_; }
    size_t get_kernel_width() const { return kernel_width_; }
};

} // namespace dlvk
