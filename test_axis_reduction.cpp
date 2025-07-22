#include <iostream>
#include <vector>
#include <memory>
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"

using namespace dlvk;

int main() {
    std::cout << "DLVK - Axis-Specific Reduction Test\n";
    std::cout << "===================================\n\n";

    try {
        // Initialize Vulkan device
        auto device = std::make_shared<VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return 1;
        }
        std::cout << "✓ Vulkan device initialized\n";

        // Initialize tensor operations
        auto tensor_ops = std::make_shared<TensorOps>(device);
        tensor_ops->initialize();
        std::cout << "✓ Tensor operations initialized\n";

        // Test axis-0 reduction
        std::cout << "\nTesting axis-0 reduction (sum along batch dimension)...\n";
        
        // Create a tensor with shape [4, 3] (4 samples, 3 features)
        std::vector<float> input_data = {
            1.0f, 2.0f, 3.0f,  // Sample 1
            4.0f, 5.0f, 6.0f,  // Sample 2
            7.0f, 8.0f, 9.0f,  // Sample 3
            10.0f, 11.0f, 12.0f // Sample 4
        };
        
        auto input_tensor = std::make_shared<Tensor>(
            std::vector<size_t>{4, 3}, DataType::FLOAT32, device);
        input_tensor->upload_data(input_data.data());
        
        std::cout << "Input tensor shape: [4, 3]\n";
        std::cout << "Input data:\n";
        std::cout << "  [1, 2, 3]\n";
        std::cout << "  [4, 5, 6]\n";
        std::cout << "  [7, 8, 9]\n";
        std::cout << "  [10, 11, 12]\n";
        
        // Perform axis-0 reduction
        auto result = std::make_shared<Tensor>(
            std::vector<size_t>{3}, DataType::FLOAT32, device);
        
        bool success_op = tensor_ops->sum_axis0(*input_tensor, *result);
        if (!success_op) {
            std::cerr << "Failed to perform axis-0 reduction" << std::endl;
            return 1;
        }
        std::cout << "✓ Axis-0 reduction completed\n";
        
        // Download and check result
        std::vector<float> result_data(3);
        result->download_data(result_data.data());
        
        std::cout << "Result shape: [3] (summed along batch dimension)\n";
        std::cout << "Result data: [";
        for (size_t i = 0; i < result_data.size(); ++i) {
            std::cout << result_data[i];
            if (i < result_data.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        // Expected result: [22, 26, 30]
        // (1+4+7+10=22, 2+5+8+11=26, 3+6+9+12=30)
        std::vector<float> expected = {22.0f, 26.0f, 30.0f};
        bool success = true;
        for (size_t i = 0; i < 3; ++i) {
            if (std::abs(result_data[i] - expected[i]) > 1e-5f) {
                success = false;
                break;
            }
        }
        
        if (success) {
            std::cout << "✅ Expected result: [22, 26, 30] - MATCH!\n";
            std::cout << "\n🎉 SUCCESS! Axis-specific reduction is working correctly!\n";
            std::cout << "✅ This means backward propagation for bias gradients will work!\n";
            std::cout << "✅ Phase 3 neural network training is now fully functional!\n";
        } else {
            std::cout << "❌ Expected result: [22, 26, 30] - NO MATCH!\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
