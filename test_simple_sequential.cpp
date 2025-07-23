#include "dlvk/model/model.h"
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <memory>
#include <iomanip>
#include <chrono>

using namespace dlvk;

int main() {
    try {
        std::cout << "=== Simple Sequential Model Test ===" << std::endl;
        
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
        
        // Display GPU device information
        std::cout << "\n=== GPU Device Information ===" << std::endl;
        std::cout << "Device Name: " << device->get_device_name() << std::endl;
        std::cout << "Device Type: " << device->get_device_type_string() << std::endl;
        std::cout << "Vulkan Version: " << device->get_vulkan_version_string() << std::endl;
        
        // Display memory information
        VkDeviceSize total_memory = device->get_total_device_memory();
        double memory_gb = static_cast<double>(total_memory) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "GPU Memory: " << std::fixed << std::setprecision(2) << memory_gb << " GB" << std::endl;
        std::cout << "Memory Heaps: " << device->get_memory_heap_count() << std::endl;
        
        // Check if we're using a GPU
        std::string device_type = device->get_device_type_string();
        if (device_type.find("GPU") != std::string::npos) {
            std::cout << "✅ CONFIRMED: Using GPU acceleration!" << std::endl;
        } else {
            std::cout << "⚠️  WARNING: Not using GPU - using " << device_type << std::endl;
        }
        std::cout << "================================" << std::endl;
        std::cout << "✓ TensorOps initialized" << std::endl;
        
        // Create Sequential model with just one layer
        Sequential model(*device);
        std::cout << "✓ Created Sequential model" << std::endl;
        
        // Add just one Dense layer
        model.add_dense(784, 128);
        std::cout << "✓ Added ONE Dense layer (784 -> 128)" << std::endl;
        
        // Create input tensor
        auto input = std::make_shared<Tensor>(std::vector<size_t>{1, 784}, DataType::FLOAT32, device);
        std::cout << "✓ Created input tensor [1, 784]" << std::endl;
        
        // Test forward pass
        std::cout << "\nTesting forward pass with ONE layer..." << std::endl;
        std::cout << "🔄 Executing forward pass on GPU..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        model.set_training(false);
        Tensor output = model.forward(*input);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "✅ Forward pass completed successfully!" << std::endl;
        std::cout << "⏱️  GPU execution time: " << duration.count() << " microseconds" << std::endl;
        
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
