#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/layers/dense_layer.h"
#include <iostream>
#include <memory>

using namespace dlvk;

int main() {
    try {
        std::cout << "=== Direct DenseLayer Test ===" << std::endl;
        
        // Initialize Vulkan device
        auto device = std::make_shared<VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize VulkanDevice" << std::endl;
            return 1;
        }
        std::cout << "✓ VulkanDevice created and initialized" << std::endl;
        
        // Initialize TensorOps singleton
        if (!TensorOps::initialize(device.get())) {
            std::cerr << "Failed to initialize TensorOps" << std::endl;
            return 1;
        }
        
        // Set the TensorOps instance for Tensor class
        auto tensor_ops = TensorOps::instance();
        Tensor::set_tensor_ops(std::shared_ptr<TensorOps>(tensor_ops, [](TensorOps*){}));
        std::cout << "✓ TensorOps initialized" << std::endl;
        
        // Create a simple tensor
        std::cout << "Creating input tensor [1, 784]..." << std::endl;
        auto input_tensor = std::make_shared<Tensor>(std::vector<size_t>{1, 784}, DataType::FLOAT32, device);
        std::cout << "✓ Input tensor created" << std::endl;
        
        // Fill it with some data
        auto ops_instance = TensorOps::instance();
        ops_instance->fill(*input_tensor, 0.5f);
        std::cout << "✓ Input tensor filled" << std::endl;
        
        // Create DenseLayer directly
        std::cout << "Creating DenseLayer (784 -> 128)..." << std::endl;
        DenseLayer dense_layer(*device, 784, 128);
        std::cout << "✓ DenseLayer created" << std::endl;
        
        // Test forward pass
        std::cout << "Testing DenseLayer forward pass..." << std::endl;
        auto output = dense_layer.forward(input_tensor);
        std::cout << "✓ Forward pass completed successfully!" << std::endl;
        
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
