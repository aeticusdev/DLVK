#include "dlvk/model/model.h"
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <memory>

using namespace dlvk;

int main() {
    std::cout << "=== Sequential Model Test ===" << std::endl;
    
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
        
        // Set TensorOps instance for Tensor static operations
        auto tensor_ops = std::make_shared<TensorOps>(device);
        Tensor::set_tensor_ops(tensor_ops);
        
        std::cout << "Vulkan device initialized successfully" << std::endl;
        
        // Create Sequential model
        Sequential model(*device);
        std::cout << "Created Sequential model" << std::endl;
        
        // Test adding layers
        try {
            // Add a dense layer using the adapter
            model.add_dense(784, 128);
            std::cout << "✓ Added Dense layer (784 -> 128)" << std::endl;
            
            // Add activation layers
            model.add_relu();
            std::cout << "✓ Added ReLU activation" << std::endl;
            
            model.add_dense(128, 10);
            std::cout << "✓ Added Dense layer (128 -> 10)" << std::endl;
            
            model.add_softmax();
            std::cout << "✓ Added Softmax activation" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Layer addition failed: " << e.what() << std::endl;
        }
        
        // Test model summary
        try {
            std::cout << "\nModel Summary:" << std::endl;
            model.summary();
            std::cout << "✓ Model summary generated successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Model summary failed: " << e.what() << std::endl;
        }
        
        // Test forward pass (if possible)
        try {
            std::cout << "\nTesting forward pass..." << std::endl;
            
            // Create a simple input tensor
            std::vector<size_t> input_shape = {1, 784};  // Batch size 1, 784 features
            Tensor input(input_shape, DataType::FLOAT32, device);
            std::cout << "Created input tensor [1, 784]" << std::endl;
            
            // Attempt forward pass
            model.set_training(false);
            Tensor output = model.forward(input);
            std::cout << "✓ Forward pass completed successfully!" << std::endl;
            std::cout << "Output shape: [";
            for (size_t i = 0; i < output.shape().size(); ++i) {
                std::cout << output.shape()[i];
                if (i < output.shape().size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Forward pass failed: " << e.what() << std::endl;
        }
        
        std::cout << "\nSequential model test completed!" << std::endl;
        
        // Cleanup
        TensorOps::shutdown();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        TensorOps::shutdown();
        return 1;
    }
}
