#include "dlvk/layers/conv2d_layer.h"
#include "dlvk/tensor/tensor_ops.h"
#include <random>
#include <cmath>

namespace dlvk {

Conv2DLayer::Conv2DLayer(VulkanDevice& device, 
                         size_t in_channels, size_t out_channels,
                         size_t kernel_height, size_t kernel_width,
                         size_t stride_h, size_t stride_w,
                         size_t padding_h, size_t padding_w)
    : device_(device), in_channels_(in_channels), out_channels_(out_channels),
      kernel_height_(kernel_height), kernel_width_(kernel_width),
      stride_h_(stride_h), stride_w_(stride_w),
      padding_h_(padding_h), padding_w_(padding_w) {
    
    // Create weight tensor: [out_channels, in_channels, kernel_h, kernel_w]
    std::vector<size_t> weight_shape = {out_channels, in_channels, kernel_height, kernel_width};
    weights_ = std::make_shared<Tensor>(weight_shape, DataType::FLOAT32, 
                                       std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    // Create bias tensor: [out_channels]
    std::vector<size_t> bias_shape = {out_channels};
    bias_ = std::make_shared<Tensor>(bias_shape, DataType::FLOAT32,
                                    std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    initialize_weights();
}

void Conv2DLayer::initialize_weights() {
    // Xavier/Glorot initialization for conv layers
    size_t fan_in = in_channels_ * kernel_height_ * kernel_width_;
    size_t fan_out = out_channels_ * kernel_height_ * kernel_width_;
    float variance = 2.0f / (fan_in + fan_out);
    float std_dev = std::sqrt(variance);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    // Initialize weights
    std::vector<float> weight_data(weights_->size());
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = dist(gen);
    }
    weights_->upload_data(weight_data.data());
    
    // Initialize bias to zero
    std::vector<float> bias_data(bias_->size(), 0.0f);
    bias_->upload_data(bias_data.data());
}

std::vector<size_t> Conv2DLayer::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // Input shape: [batch_size, in_channels, height, width]
    // Output shape: [batch_size, out_channels, out_height, out_width]
    
    if (input_shape.size() != 4) {
        throw std::runtime_error("Conv2D input must be 4D: [batch, channels, height, width]");
    }
    
    size_t batch_size = input_shape[0];
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];
    
    size_t output_height = (input_height + 2 * padding_h_ - kernel_height_) / stride_h_ + 1;
    size_t output_width = (input_width + 2 * padding_w_ - kernel_width_) / stride_w_ + 1;
    
    return {batch_size, out_channels_, output_height, output_width};
}

std::shared_ptr<Tensor> Conv2DLayer::forward(const std::shared_ptr<Tensor>& input) {
    // Store input for backward pass
    last_input_ = input;
    
    // For now, implement a basic CPU-based convolution
    // TODO: Implement GPU compute shader for convolution
    
    auto output_shape = compute_output_shape(input->shape());
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32,
                                          std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    // Download input and weights for CPU computation
    std::vector<float> input_data(input->size());
    std::vector<float> weight_data(weights_->size());
    std::vector<float> bias_data(bias_->size());
    
    input->download_data(input_data.data());
    weights_->download_data(weight_data.data());
    bias_->download_data(bias_data.data());
    
    // CPU implementation of convolution
    const auto& in_shape = input->shape();
    size_t batch_size = in_shape[0];
    size_t input_height = in_shape[2];
    size_t input_width = in_shape[3];
    size_t output_height = output_shape[2];
    size_t output_width = output_shape[3];
    
    std::vector<float> output_data(output->size(), 0.0f);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    float sum = bias_data[oc]; // Add bias
                    
                    for (size_t ic = 0; ic < in_channels_; ++ic) {
                        for (size_t kh = 0; kh < kernel_height_; ++kh) {
                            for (size_t kw = 0; kw < kernel_width_; ++kw) {
                                int ih = static_cast<int>(oh * stride_h_ + kh) - static_cast<int>(padding_h_);
                                int iw = static_cast<int>(ow * stride_w_ + kw) - static_cast<int>(padding_w_);
                                
                                if (ih >= 0 && ih < static_cast<int>(input_height) && 
                                    iw >= 0 && iw < static_cast<int>(input_width)) {
                                    
                                    size_t input_idx = b * (in_channels_ * input_height * input_width) +
                                                      ic * (input_height * input_width) +
                                                      ih * input_width + iw;
                                    
                                    size_t weight_idx = oc * (in_channels_ * kernel_height_ * kernel_width_) +
                                                       ic * (kernel_height_ * kernel_width_) +
                                                       kh * kernel_width_ + kw;
                                    
                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                    
                    size_t output_idx = b * (out_channels_ * output_height * output_width) +
                                       oc * (output_height * output_width) +
                                       oh * output_width + ow;
                    output_data[output_idx] = sum;
                }
            }
        }
    }
    
    output->upload_data(output_data.data());
    return output;
}

std::shared_ptr<Tensor> Conv2DLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    // For now, implement basic CPU-based backward pass
    // TODO: Implement GPU compute shader for convolution backward
    
    if (!last_input_) {
        throw std::runtime_error("Conv2DLayer::backward called without prior forward pass");
    }
    
    auto input_shape = last_input_->shape();
    auto grad_input = std::make_shared<Tensor>(input_shape, DataType::FLOAT32,
                                              std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    // Initialize gradient input to zero
    std::vector<float> grad_input_data(grad_input->size(), 0.0f);
    grad_input->upload_data(grad_input_data.data());
    
    // Note: For a complete implementation, we would:
    // 1. Compute gradients w.r.t. input (for backpropagation)
    // 2. Compute gradients w.r.t. weights (for weight updates)
    // 3. Compute gradients w.r.t. bias (for bias updates)
    
    // For now, return zero gradients (placeholder)
    return grad_input;
}

void Conv2DLayer::update_weights(float learning_rate) {
    // For now, implement basic weight update
    // TODO: Implement proper gradient-based weight updates
    
    // This is a placeholder - in a complete implementation,
    // we would apply the computed gradients to weights and bias
    // weights_ = weights_ - learning_rate * weight_gradients
    // bias_ = bias_ - learning_rate * bias_gradients
}

} // namespace dlvk
