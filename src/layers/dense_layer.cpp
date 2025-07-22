#include "dlvk/layers/layer.h"
#include "dlvk/core/vulkan_device.h"
#include <random>

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
    // Compute gradients
    // grad_input = grad_output @ weights.T
    // grad_weights = input.T @ grad_output
    // grad_bias = sum(grad_output, axis=0)
    
    // TODO: Implement proper backward pass with compute shaders
    
    // For now, return a placeholder
    return std::make_shared<Tensor>(m_last_input->shape(), DataType::FLOAT32, m_device);
}

void DenseLayer::update_weights(float learning_rate) {
    // TODO: Implement weight update using computed gradients
    // weights -= learning_rate * grad_weights
    // bias -= learning_rate * grad_bias
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
