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
    std::cout << "=== Sequential Model Test ===" << std::endl;
    
    try {
        // Create Vulkan device
        auto device = std::make_shared<VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return 1;
        }
        
        // Initialize TensorOps singleton
        if (!TensorOps::initialize(device.get())) {
            std::cerr << "Failed to initialize TensorOps" << std::endl;
            return 1;
        }
        
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
        
        // Use the singleton instance for all operations
        auto tensor_ops_instance = TensorOps::instance();
        if (!tensor_ops_instance) {
            std::cerr << "Failed to get TensorOps singleton instance" << std::endl;
            return 1;
        }
        
        // Create a shared_ptr wrapper for the singleton (without owning it)
        std::shared_ptr<TensorOps> tensor_ops(tensor_ops_instance, [](TensorOps*) {
            // Custom deleter that does nothing since we don't own the singleton
        });
        Tensor::set_tensor_ops(tensor_ops);
        
        std::cout << "Vulkan device initialized successfully" << std::endl;
        
        // Create Sequential model
        Sequential model(device);
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
            auto input = std::make_shared<Tensor>(input_shape, DataType::FLOAT32, device);
            std::cout << "Created input tensor [1, 784]" << std::endl;
            
            // Attempt forward pass
            model.set_training(false);
            
            std::cout << "🔄 Executing forward pass on GPU..." << std::endl;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            Tensor output = model.forward(*input);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "✅ Forward pass completed successfully!" << std::endl;
            std::cout << "⏱️  GPU execution time: " << duration.count() << " microseconds" << std::endl;
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
        
        // Clear the static tensor ops pointer before cleanup
        Tensor::set_tensor_ops(nullptr);
        
        // Skip explicit shutdown to test if that's causing the issue
        // TensorOps::shutdown();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        
        // Clear the static tensor ops pointer before cleanup
        Tensor::set_tensor_ops(nullptr);
        // TensorOps::shutdown();
        return 1;
    }
}
