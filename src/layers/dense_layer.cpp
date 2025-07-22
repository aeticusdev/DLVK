#include "dlvk/layers/layer.h"
#include "dlvk/core/vulkan_device.h"
#include <random>
#include <iostream>

namespace dlvk {

DenseLayer::DenseLayer(size_t input_size, size_t output_size, std::shared_ptr<VulkanDevice> device)
    : m_input_size(input_size), m_output_size(output_size) {
    m_device = device;
    
    // Initialize weight and bias tensors
    m_weights = std::make_shared<Tensor>(
        std::vector<size_t>{input_size, output_size}, 
        DataType::FLOAT32, 
        device
    );
    
    m_bias = std::make_shared<Tensor>(
        std::vector<size_t>{output_size}, 
        DataType::FLOAT32, 
        device
    );
    
    initialize_weights();
}

std::shared_ptr<Tensor> DenseLayer::forward(const std::shared_ptr<Tensor>& input) {
    // Store input for backward pass
    m_last_input = input;
    
    // Perform matrix multiplication: input @ weights
    auto output = input->matrix_multiply(*m_weights);
    
    // Add bias with broadcasting: output + bias
    auto result = output->add_broadcast(*m_bias);
    
    return result;
}

std::shared_ptr<Tensor> DenseLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    if (!m_last_input) {
        throw std::runtime_error("No forward pass recorded for backward pass");
    }
    
    // Compute gradients
    // grad_input = grad_output @ weights.T
    // grad_weights = input.T @ grad_output  
    // grad_bias = sum(grad_output, axis=0)
    
    // 1. Compute gradient w.r.t input: grad_input = grad_output @ weights.T
    auto weights_T = m_weights->transpose();
    auto grad_input = grad_output->matrix_multiply(*weights_T);
    
    // 2. Compute gradient w.r.t weights: grad_weights = input.T @ grad_output
    auto input_T = m_last_input->transpose();
    m_grad_weights = input_T->matrix_multiply(*grad_output);
    
    // 3. Compute gradient w.r.t bias: grad_bias = sum(grad_output, axis=0)
    m_grad_bias = grad_output->sum(0);  // Sum along batch dimension
    
    return grad_input;
}

void DenseLayer::update_weights(float learning_rate) {
    if (!m_grad_weights || !m_grad_bias) {
        std::cerr << "Warning: No gradients computed for weight update" << std::endl;
        return;
    }
    
    // Update weights: weights -= learning_rate * grad_weights
    auto lr_grad_weights = m_grad_weights->multiply_scalar(-learning_rate);
    m_weights = m_weights->add(*lr_grad_weights);
    
    // Update bias: bias -= learning_rate * grad_bias  
    auto lr_grad_bias = m_grad_bias->multiply_scalar(-learning_rate);
    m_bias = m_bias->add(*lr_grad_bias);
}

void DenseLayer::initialize_weights() {
    // Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (m_input_size + m_output_size));
    std::normal_distribution<float> dis(0.0f, scale);
    
    // Initialize weights
    std::vector<float> weight_data(m_input_size * m_output_size);
    for (auto& w : weight_data) {
        w = dis(gen);
    }
    m_weights->upload_data(weight_data.data());
    
    // Initialize bias to zero
    std::vector<float> bias_data(m_output_size, 0.0f);
    m_bias->upload_data(bias_data.data());
}

} // namespace dlvk
