#include "dlvk/layers/dropout_layer.h"
#include <algorithm>
#include <stdexcept>

namespace dlvk {

DropoutLayer::DropoutLayer(VulkanDevice& device, float dropout_rate)
    : device_(device), dropout_rate_(dropout_rate), training_(true),
      generator_(std::random_device{}()), distribution_(0.0f, 1.0f) {
    
    if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in range [0, 1)");
    }
}

std::shared_ptr<Tensor> DropoutLayer::forward(const std::shared_ptr<Tensor>& input) {
    if (!training_ || dropout_rate_ == 0.0f) {
        // During inference or with 0 dropout, return input unchanged
        return input;
    }
    
    // Download input data
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    // Create dropout mask and apply it
    std::vector<float> mask_data(input->size());
    std::vector<float> output_data(input->size());
    
    float scale = 1.0f / (1.0f - dropout_rate_);  // Inverted dropout scaling
    
    for (size_t i = 0; i < input->size(); ++i) {
        bool keep = distribution_(generator_) >= dropout_rate_;
        mask_data[i] = keep ? scale : 0.0f;
        output_data[i] = input_data[i] * mask_data[i];
    }
    
    // Store mask for backward pass
    mask_ = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32,
                                    std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    mask_->upload_data(mask_data.data());
    
    // Create output tensor
    auto output = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32,
                                          std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    output->upload_data(output_data.data());
    
    return output;
}

std::shared_ptr<Tensor> DropoutLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    if (!training_ || dropout_rate_ == 0.0f || !mask_) {
        // During inference or with 0 dropout, return gradient unchanged
        return grad_output;
    }
    
    // Apply the same mask to the gradient
    return grad_output->multiply(*mask_);
}

std::unique_ptr<Layer> DropoutLayer::clone() const {
    auto cloned = std::make_unique<DropoutLayer>(device_, dropout_rate_);
    cloned->training_ = training_;
    return cloned;
}

} // namespace dlvk
