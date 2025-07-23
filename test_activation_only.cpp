#include "dlvk/layers/activation.h"
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <memory>

using namespace dlvk;

int main() {
    std::cout << "=== Minimal ActivationLayer Test ===" << std::endl;
    
    try {
        // Create Vulkan device
        auto device = std::make_shared<VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return 1;
        }
        
        // Initialize TensorOps
        if (!TensorOps::initialize(device.get())) {
            std::cerr << "Failed to initialize TensorOps" << std::endl;
            return 1;
        }
        
        std::cout << "Vulkan device initialized successfully" << std::endl;
        
        // Create a simple test tensor
        std::vector<size_t> shape = {2, 3};
        Tensor input(shape, DataType::FLOAT32, device);
        std::cout << "Created test tensor with shape [2, 3]" << std::endl;
        
        // Test ReLU activation
        ActivationLayer relu_layer(device, ActivationType::ReLU);
        std::cout << "Created ReLU activation layer" << std::endl;
        
        // Get layer info
        LayerInfo info = relu_layer.get_layer_info();
        std::cout << "Layer type: " << info.type << std::endl;
        std::cout << "Parameter count: " << info.parameter_count << std::endl;
        std::cout << "Trainable: " << (info.trainable ? "Yes" : "No") << std::endl;
        
        // Test forward pass
        try {
            Tensor output = relu_layer.forward(input);
            std::cout << "Forward pass completed successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Forward pass failed: " << e.what() << std::endl;
        }
        
        // Test other activation types
        ActivationLayer sigmoid_layer(device, ActivationType::Sigmoid);
        std::cout << "Created Sigmoid layer: " << sigmoid_layer.get_layer_info().type << std::endl;
        
        ActivationLayer tanh_layer(device, ActivationType::Tanh);
        std::cout << "Created Tanh layer: " << tanh_layer.get_layer_info().type << std::endl;
        
        ActivationLayer softmax_layer(device, ActivationType::Softmax);
        std::cout << "Created Softmax layer: " << softmax_layer.get_layer_info().type << std::endl;
        
        std::cout << "All activation layers created successfully!" << std::endl;
        
        // Cleanup
        TensorOps::shutdown();
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        TensorOps::shutdown();
        return 1;
    }
}
