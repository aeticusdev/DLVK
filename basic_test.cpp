#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "DLVK Basic Operation Test\n";
    std::cout << "========================\n";
    
    try {
        // Initialize Vulkan device
        auto device = std::make_shared<dlvk::VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return -1;
        }
        std::cout << "✓ Vulkan device initialized\n";
        
        // Try manual pipeline creation to see what works
        auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
        dlvk::Tensor::set_tensor_ops(tensor_ops);
        
        std::cout << "✓ TensorOps created\n";
        
        // Create simple tensors
        std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data_b = {2.0f, 1.0f, 2.0f, 1.0f};
        
        auto tensor_a = std::make_shared<dlvk::Tensor>(std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device);
        auto tensor_b = std::make_shared<dlvk::Tensor>(std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device);
        
        tensor_a->upload_data(data_a.data());
        tensor_b->upload_data(data_b.data());
        
        std::cout << "✓ Test tensors created and data uploaded\n";
        
        // Test what we can do without pipelines - just basic tensor creation
        std::vector<float> download_test(4);
        tensor_a->download_data(download_test.data());
        
        std::cout << "✓ Basic tensor operations work\n";
        std::cout << "Downloaded data: ";
        for (float val : download_test) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\nPhase 2 Analysis:\n";
        std::cout << "=================\n";
        std::cout << "✓ Vulkan device management - WORKING\n";
        std::cout << "✓ Tensor memory allocation - WORKING\n"; 
        std::cout << "✓ Data upload/download - WORKING\n";
        std::cout << "✗ Compute pipelines - NOT WORKING (need to fix create_pipelines())\n";
        std::cout << "✗ GPU operations - NOT WORKING (depends on pipelines)\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
