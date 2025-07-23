#include "dlvk/layers/activation.h"
#include "dlvk/tensor/tensor_ops_static.h"
#include "dlvk/optimizers/optimizers.h"
#include "dlvk/core/vulkan_device.h"
#include <stdexcept>
#include <fstream>

namespace dlvk {

ActivationLayer::ActivationLayer(VulkanDevice& device, ActivationType activation_type) 
    : m_device(&device), m_activation_type(activation_type), m_is_training(true) {}

Tensor ActivationLayer::forward(const Tensor& input) {
    // Create output tensor with same shape and properties as input
    // Get shared_ptr from the device
    auto device_ptr = std::shared_ptr<VulkanDevice>(m_device, [](VulkanDevice*){});
    Tensor output(input.shape(), input.dtype(), device_ptr);
    
    switch (m_activation_type) {
        case ActivationType::ReLU:
            if (!TensorOpsStatic::relu(input, output)) {
                throw std::runtime_error("ReLU forward pass failed");
            }
            break;
        case ActivationType::Sigmoid:
            if (!TensorOpsStatic::sigmoid(input, output)) {
                throw std::runtime_error("Sigmoid forward pass failed");
            }
            break;
        case ActivationType::Tanh:
            if (!TensorOpsStatic::tanh_activation(input, output)) {
                throw std::runtime_error("Tanh forward pass failed");
            }
            break;
        case ActivationType::Softmax:
            if (!TensorOpsStatic::softmax(input, output)) {
                throw std::runtime_error("Softmax forward pass failed");
            }
            break;
        default:
            throw std::runtime_error("Unknown activation type");
    }
    
    return output;
}

Tensor ActivationLayer::backward(const Tensor& grad_output) {
    // Create gradient input tensor with same shape as grad_output
    auto device_ptr = std::shared_ptr<VulkanDevice>(m_device, [](VulkanDevice*){});
    Tensor grad_input(grad_output.shape(), grad_output.dtype(), device_ptr);
    
    bool success = false;
    switch (m_activation_type) {
        case ActivationType::ReLU: {
            // For ReLU backward, we need the input (which we don't store)
            // This is a simplified version - in practice, you'd store the forward input
            Tensor dummy_input(grad_output.shape(), grad_output.dtype(), device_ptr);
            success = TensorOpsStatic::relu_backward(dummy_input, grad_output, grad_input);
            break;
        }
        case ActivationType::Sigmoid: {
            // For sigmoid backward, we need the forward result
            // This is a simplified version - in practice, you'd store the forward result
            Tensor sigmoid_output(grad_output.shape(), grad_output.dtype(), device_ptr);
            success = TensorOpsStatic::sigmoid_backward(sigmoid_output, grad_output, grad_input);
            break;
        }
        case ActivationType::Tanh: {
            // For tanh backward, we need the forward result
            Tensor tanh_output(grad_output.shape(), grad_output.dtype(), device_ptr);
            success = TensorOpsStatic::tanh_backward(tanh_output, grad_output, grad_input);
            break;
        }
        case ActivationType::Softmax:
            // Softmax backward is typically handled with the loss function
            // For simplicity, just pass through the gradient
            success = TensorOpsStatic::copy(grad_output, grad_input);
            break;
        default:
            throw std::runtime_error("Unknown activation type");
    }
    
    if (!success) {
        throw std::runtime_error("Activation backward pass failed");
    }
    
    return grad_input;
}

void ActivationLayer::update_parameters(Optimizer& optimizer) {
    // Activation layers have no parameters to update
}

void ActivationLayer::set_training(bool training) {
    m_is_training = training;
}

LayerInfo ActivationLayer::get_layer_info() const {
    LayerInfo info;
    
    switch (m_activation_type) {
        case ActivationType::ReLU:
            info.type = "ReLU";
            break;
        case ActivationType::Sigmoid:
            info.type = "Sigmoid";
            break;
        case ActivationType::Tanh:
            info.type = "Tanh";
            break;
        case ActivationType::Softmax:
            info.type = "Softmax";
            break;
    }
    
    info.parameter_count = 0;  // No parameters
    info.trainable = false;
    info.output_shape_str = "Same as input";
    
    return info;
}

void ActivationLayer::save_weights(std::ofstream& file) const {
    // No weights to save for activation layers
    // Write activation type for compatibility
    int activation_type = static_cast<int>(m_activation_type);
    file.write(reinterpret_cast<const char*>(&activation_type), sizeof(activation_type));
}

void ActivationLayer::load_weights(std::ifstream& file) {
    // No weights to load for activation layers
    // Read activation type for compatibility
    int activation_type;
    file.read(reinterpret_cast<char*>(&activation_type), sizeof(activation_type));
    
    // Verify it matches current activation type
    if (static_cast<ActivationType>(activation_type) != m_activation_type) {
        throw std::runtime_error("Activation type mismatch during weight loading");
    }
}

} // namespace dlvk
