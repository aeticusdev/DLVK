#include "dlvk/tensor/tensor_ops_static.h"
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include <iostream>

using namespace dlvk;

int main() {
    try {
        std::cout << "=== Matrix Multiply Pipeline Test ===" << std::endl;
        
        // Initialize Vulkan device
        auto device = std::make_shared<VulkanDevice>();
        
        // TensorOpsStatic should auto-initialize when needed
        std::cout << "✓ TensorOpsStatic ready to use" << std::endl;
        
        // Create test tensors for matrix multiply: (2,3) x (3,2) = (2,2)
        auto tensor_a = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, DataType::FLOAT32, device);
        auto tensor_b = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, DataType::FLOAT32, device);
        auto result = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, DataType::FLOAT32, device);
        
        // Fill with test data
        std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> data_b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        
        tensor_a->upload_data(data_a.data());
        tensor_b->upload_data(data_b.data());
        
        std::cout << "✓ Test tensors created and filled" << std::endl;
        
        // Perform matrix multiplication
        std::cout << "Attempting matrix multiply..." << std::endl;
        bool success = TensorOpsStatic::matrix_multiply(*tensor_a, *tensor_b, *result);
        
        if (success) {
            std::cout << "✅ Matrix multiply succeeded!" << std::endl;
            
            // Download and print result
            std::vector<float> result_data(4);
            result->download_data(result_data.data());
            
            std::cout << "Result matrix (2x2):" << std::endl;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    std::cout << result_data[i*2 + j] << " ";
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "❌ Matrix multiply failed!" << std::endl;
        }
        
        return success ? 0 : -1;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
}
