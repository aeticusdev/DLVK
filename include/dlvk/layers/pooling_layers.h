#pragma once

#include "layer.h"
#include "../tensor/tensor.h"
#include <memory>

namespace dlvk {

class MaxPool2DLayer : public Layer {
private:
    std::shared_ptr<Tensor> last_input_;     // For backward pass
    std::shared_ptr<Tensor> max_indices_;    // Store max indices for backward pass
    
    size_t pool_height_;
    size_t pool_width_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;
    VulkanDevice& device_;

public:
    MaxPool2DLayer(VulkanDevice& device,
                   size_t pool_height, size_t pool_width,
                   size_t stride_h = 1, size_t stride_w = 1,
                   size_t padding_h = 0, size_t padding_w = 0);
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    void update_weights(float learning_rate) override {} // No weights to update
    std::unique_ptr<Layer> clone() const override;
    

    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const;
    

    size_t get_pool_height() const { return pool_height_; }
    size_t get_pool_width() const { return pool_width_; }
};

class AvgPool2DLayer : public Layer {
private:
    std::shared_ptr<Tensor> last_input_;     // For backward pass
    
    size_t pool_height_;
    size_t pool_width_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;
    VulkanDevice& device_;

public:
    AvgPool2DLayer(VulkanDevice& device,
                   size_t pool_height, size_t pool_width,
                   size_t stride_h = 1, size_t stride_w = 1,
                   size_t padding_h = 0, size_t padding_w = 0);
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    void update_weights(float learning_rate) override {} // No weights to update
    std::unique_ptr<Layer> clone() const override;
    

    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const;
    

    size_t get_pool_height() const { return pool_height_; }
    size_t get_pool_width() const { return pool_width_; }
};

} // namespace dlvk
