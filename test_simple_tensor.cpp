#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <memory>

using namespace dlvk;

int main() {
    try {
        std::cout << "=== Simple Tensor Creation Test ===" << std::endl;
        
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
        std::cout << "✓ TensorOps initialized" << std::endl;
        
        // Test creating a simple tensor
        std::cout << "Creating tensor [1, 784]..." << std::endl;
        Tensor input_tensor({1, 784}, DataType::FLOAT32, device);
        std::cout << "✓ Tensor created successfully" << std::endl;
        
        // Test filling the tensor using TensorOps
        std::cout << "Filling tensor with value 0.5..." << std::endl;
        auto tensor_ops = TensorOps::instance();
        tensor_ops->fill(input_tensor, 0.5f);
        std::cout << "✓ Tensor filled successfully" << std::endl;
        
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
