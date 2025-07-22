#include "dlvk/layers/pooling_layers.h"
#include <limits>
#include <algorithm>

namespace dlvk {

// MaxPool2D Implementation
MaxPool2DLayer::MaxPool2DLayer(VulkanDevice& device,
                               size_t pool_height, size_t pool_width,
                               size_t stride_h, size_t stride_w,
                               size_t padding_h, size_t padding_w)
    : device_(device), pool_height_(pool_height), pool_width_(pool_width),
      stride_h_(stride_h), stride_w_(stride_w),
      padding_h_(padding_h), padding_w_(padding_w) {
}

std::vector<size_t> MaxPool2DLayer::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // Input shape: [batch_size, channels, height, width]
    // Output shape: [batch_size, channels, out_height, out_width]
    
    if (input_shape.size() != 4) {
        throw std::runtime_error("MaxPool2D input must be 4D: [batch, channels, height, width]");
    }
    
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];
    
    size_t output_height = (input_height + 2 * padding_h_ - pool_height_) / stride_h_ + 1;
    size_t output_width = (input_width + 2 * padding_w_ - pool_width_) / stride_w_ + 1;
    
    return {batch_size, channels, output_height, output_width};
}

std::shared_ptr<Tensor> MaxPool2DLayer::forward(const std::shared_ptr<Tensor>& input) {
    last_input_ = input;
    
    auto output_shape = compute_output_shape(input->shape());
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32,
                                          std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    // Store indices for backward pass
    max_indices_ = std::make_shared<Tensor>(output_shape, DataType::FLOAT32,
                                           std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    // Download input for CPU computation
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    const auto& in_shape = input->shape();
    size_t batch_size = in_shape[0];
    size_t channels = in_shape[1];
    size_t input_height = in_shape[2];
    size_t input_width = in_shape[3];
    size_t output_height = output_shape[2];
    size_t output_width = output_shape[3];
    
    std::vector<float> output_data(output->size());
    std::vector<float> indices_data(max_indices_->size());
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    size_t max_idx = 0;
                    
                    for (size_t ph = 0; ph < pool_height_; ++ph) {
                        for (size_t pw = 0; pw < pool_width_; ++pw) {
                            int ih = static_cast<int>(oh * stride_h_ + ph) - static_cast<int>(padding_h_);
                            int iw = static_cast<int>(ow * stride_w_ + pw) - static_cast<int>(padding_w_);
                            
                            if (ih >= 0 && ih < static_cast<int>(input_height) && 
                                iw >= 0 && iw < static_cast<int>(input_width)) {
                                
                                size_t input_idx = b * (channels * input_height * input_width) +
                                                  c * (input_height * input_width) +
                                                  ih * input_width + iw;
                                
                                if (input_data[input_idx] > max_val) {
                                    max_val = input_data[input_idx];
                                    max_idx = input_idx;
                                }
                            }
                        }
                    }
                    
                    size_t output_idx = b * (channels * output_height * output_width) +
                                       c * (output_height * output_width) +
                                       oh * output_width + ow;
                    
                    output_data[output_idx] = max_val;
                    indices_data[output_idx] = static_cast<float>(max_idx);
                }
            }
        }
    }
    
    output->upload_data(output_data.data());
    max_indices_->upload_data(indices_data.data());
    
    return output;
}

std::shared_ptr<Tensor> MaxPool2DLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    if (!last_input_ || !max_indices_) {
        throw std::runtime_error("MaxPool2DLayer::backward called without prior forward pass");
    }
    
    auto input_shape = last_input_->shape();
    auto grad_input = std::make_shared<Tensor>(input_shape, DataType::FLOAT32,
                                              std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    // Initialize gradient input to zero
    std::vector<float> grad_input_data(grad_input->size(), 0.0f);
    
    // Download gradients and indices
    std::vector<float> grad_output_data(grad_output->size());
    std::vector<float> indices_data(max_indices_->size());
    
    grad_output->download_data(grad_output_data.data());
    max_indices_->download_data(indices_data.data());
    
    // Propagate gradients back to the max positions
    for (size_t i = 0; i < grad_output_data.size(); ++i) {
        size_t max_idx = static_cast<size_t>(indices_data[i]);
        grad_input_data[max_idx] += grad_output_data[i];
    }
    
    grad_input->upload_data(grad_input_data.data());
    return grad_input;
}

// AvgPool2D Implementation
AvgPool2DLayer::AvgPool2DLayer(VulkanDevice& device,
                               size_t pool_height, size_t pool_width,
                               size_t stride_h, size_t stride_w,
                               size_t padding_h, size_t padding_w)
    : device_(device), pool_height_(pool_height), pool_width_(pool_width),
      stride_h_(stride_h), stride_w_(stride_w),
      padding_h_(padding_h), padding_w_(padding_w) {
}

std::vector<size_t> AvgPool2DLayer::compute_output_shape(const std::vector<size_t>& input_shape) const {
    if (input_shape.size() != 4) {
        throw std::runtime_error("AvgPool2D input must be 4D: [batch, channels, height, width]");
    }
    
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];
    
    size_t output_height = (input_height + 2 * padding_h_ - pool_height_) / stride_h_ + 1;
    size_t output_width = (input_width + 2 * padding_w_ - pool_width_) / stride_w_ + 1;
    
    return {batch_size, channels, output_height, output_width};
}

std::shared_ptr<Tensor> AvgPool2DLayer::forward(const std::shared_ptr<Tensor>& input) {
    last_input_ = input;
    
    auto output_shape = compute_output_shape(input->shape());
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32,
                                          std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    // Download input for CPU computation
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    const auto& in_shape = input->shape();
    size_t batch_size = in_shape[0];
    size_t channels = in_shape[1];
    size_t input_height = in_shape[2];
    size_t input_width = in_shape[3];
    size_t output_height = output_shape[2];
    size_t output_width = output_shape[3];
    
    std::vector<float> output_data(output->size());
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    float sum = 0.0f;
                    size_t count = 0;
                    
                    for (size_t ph = 0; ph < pool_height_; ++ph) {
                        for (size_t pw = 0; pw < pool_width_; ++pw) {
                            int ih = static_cast<int>(oh * stride_h_ + ph) - static_cast<int>(padding_h_);
                            int iw = static_cast<int>(ow * stride_w_ + pw) - static_cast<int>(padding_w_);
                            
                            if (ih >= 0 && ih < static_cast<int>(input_height) && 
                                iw >= 0 && iw < static_cast<int>(input_width)) {
                                
                                size_t input_idx = b * (channels * input_height * input_width) +
                                                  c * (input_height * input_width) +
                                                  ih * input_width + iw;
                                
                                sum += input_data[input_idx];
                                count++;
                            }
                        }
                    }
                    
                    size_t output_idx = b * (channels * output_height * output_width) +
                                       c * (output_height * output_width) +
                                       oh * output_width + ow;
                    
                    output_data[output_idx] = (count > 0) ? sum / count : 0.0f;
                }
            }
        }
    }
    
    output->upload_data(output_data.data());
    return output;
}

std::shared_ptr<Tensor> AvgPool2DLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    if (!last_input_) {
        throw std::runtime_error("AvgPool2DLayer::backward called without prior forward pass");
    }
    
    auto input_shape = last_input_->shape();
    auto grad_input = std::make_shared<Tensor>(input_shape, DataType::FLOAT32,
                                              std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    // Initialize gradient input to zero
    std::vector<float> grad_input_data(grad_input->size(), 0.0f);
    
    // Download gradients
    std::vector<float> grad_output_data(grad_output->size());
    grad_output->download_data(grad_output_data.data());
    
    const auto& out_shape = grad_output->shape();
    size_t batch_size = out_shape[0];
    size_t channels = out_shape[1];
    size_t output_height = out_shape[2];
    size_t output_width = out_shape[3];
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];
    
    // Distribute gradients evenly across the pooling window
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    size_t output_idx = b * (channels * output_height * output_width) +
                                       c * (output_height * output_width) +
                                       oh * output_width + ow;
                    
                    float grad_val = grad_output_data[output_idx];
                    size_t pool_size = pool_height_ * pool_width_;
                    float distributed_grad = grad_val / pool_size;
                    
                    for (size_t ph = 0; ph < pool_height_; ++ph) {
                        for (size_t pw = 0; pw < pool_width_; ++pw) {
                            int ih = static_cast<int>(oh * stride_h_ + ph) - static_cast<int>(padding_h_);
                            int iw = static_cast<int>(ow * stride_w_ + pw) - static_cast<int>(padding_w_);
                            
                            if (ih >= 0 && ih < static_cast<int>(input_height) && 
                                iw >= 0 && iw < static_cast<int>(input_width)) {
                                
                                size_t input_idx = b * (channels * input_height * input_width) +
                                                  c * (input_height * input_width) +
                                                  ih * input_width + iw;
                                
                                grad_input_data[input_idx] += distributed_grad;
                            }
                        }
                    }
                }
            }
        }
    }
    
    grad_input->upload_data(grad_input_data.data());
    return grad_input;
}

} // namespace dlvk
