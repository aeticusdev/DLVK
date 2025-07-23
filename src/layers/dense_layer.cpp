#include "dlvk/layers/dense_layer.h"
#include "dlvk/core/vulkan_device.h"
#include <random>
#include <iostream>

namespace dlvk {

DenseLayer::DenseLayer(VulkanDevice& device, size_t input_size, size_t output_size)
    : device_(device), input_size_(input_size), output_size_(output_size) {
    
    // IMPORTANT: Don't copy the VulkanDevice! Get a shared_ptr to the existing instance
    // We need to use a custom deleter that does nothing to avoid double-deletion
    auto device_ptr = std::shared_ptr<VulkanDevice>(&device, [](VulkanDevice*){});
    
    // Initialize weight and bias tensors
    weights_ = std::make_shared<Tensor>(
        std::vector<size_t>{input_size, output_size}, 
        DataType::FLOAT32, 
        device_ptr
    );
    
    bias_ = std::make_shared<Tensor>(
        std::vector<size_t>{output_size}, 
        DataType::FLOAT32, 
        device_ptr
    );
    
    initialize_weights();
}

std::shared_ptr<Tensor> DenseLayer::forward(const std::shared_ptr<Tensor>& input) {
    // Store input for backward pass
    last_input_ = input;
    
    // Perform matrix multiplication: input @ weights
    auto output = input->matrix_multiply(*weights_);
    
    // Add bias with broadcasting: output + bias
    auto result = output->add_broadcast(*bias_);
    
    return result;
}

std::shared_ptr<Tensor> DenseLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    if (!last_input_) {
        throw std::runtime_error("No forward pass recorded for backward pass");
    }
    
    // Compute gradients
    // grad_input = grad_output @ weights.T
    // grad_weights = input.T @ grad_output  
    // grad_bias = sum(grad_output, axis=0)
    
    // 1. Compute gradient w.r.t input: grad_input = grad_output @ weights.T
    auto weights_T = weights_->transpose();
    auto grad_input = grad_output->matrix_multiply(*weights_T);
    
    // 2. Compute gradient w.r.t weights: grad_weights = input.T @ grad_output
    auto input_T = last_input_->transpose();
    grad_weights_ = input_T->matrix_multiply(*grad_output);
    
    // 3. Compute gradient w.r.t bias: grad_bias = sum(grad_output, axis=0)
    grad_bias_ = grad_output->sum(0);  // Sum along batch dimension
    
    return grad_input;
}

void DenseLayer::update_weights(float learning_rate) {
    if (!grad_weights_ || !grad_bias_) {
        std::cerr << "Warning: No gradients computed for weight update" << std::endl;
        return;
    }
    
    // Update weights: weights -= learning_rate * grad_weights
    auto lr_grad_weights = grad_weights_->multiply_scalar(-learning_rate);
    weights_ = weights_->add(*lr_grad_weights);
    
    // Update bias: bias -= learning_rate * grad_bias  
    auto lr_grad_bias = grad_bias_->multiply_scalar(-learning_rate);
    bias_ = bias_->add(*lr_grad_bias);
}

void DenseLayer::initialize_weights() {
    // Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (input_size_ + output_size_));
    std::normal_distribution<float> dis(0.0f, scale);
    
    // Initialize weights
    std::vector<float> weight_data(input_size_ * output_size_);
    for (auto& w : weight_data) {
        w = dis(gen);
    }
    weights_->upload_data(weight_data.data());
    
    // Initialize bias to zero
    std::vector<float> bias_data(output_size_, 0.0f);
    bias_->upload_data(bias_data.data());
}

} // namespace dlvk
