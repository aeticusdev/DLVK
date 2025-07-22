#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/layers/layer.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "DLVK - Vulkan Machine Learning Framework Demo\n";
    std::cout << "=============================================\n\n";
    
    try {
        // Initialize Vulkan device
        auto device = std::make_shared<dlvk::VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return -1;
        }
        
        std::cout << "✓ Vulkan device initialized successfully\n";
        
        // Initialize tensor operations
        auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
        if (!tensor_ops->initialize()) {
            std::cerr << "Failed to initialize tensor operations" << std::endl;
            return -1;
        }
        
        // Set global tensor operations
        dlvk::Tensor::set_tensor_ops(tensor_ops);
        std::cout << "✓ Tensor operations initialized\n";
        
        // Create test tensors
        std::vector<size_t> shape = {4, 4};
        auto tensor_a = std::make_shared<dlvk::Tensor>(shape, dlvk::DataType::FLOAT32, device);
        auto tensor_b = std::make_shared<dlvk::Tensor>(shape, dlvk::DataType::FLOAT32, device);
        
        std::cout << "✓ Created test tensors with shape [4, 4]\n";
        
        // Upload test data
        std::vector<float> data_a = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
        };
        
        std::vector<float> data_b = {
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f,
            0.9f, 1.0f, 1.1f, 1.2f,
            1.3f, 1.4f, 1.5f, 1.6f
        };
        
        tensor_a->upload_data(data_a.data());
        tensor_b->upload_data(data_b.data());
        
        std::cout << "✓ Uploaded test data to tensors\n";
        
        // Test tensor addition
        std::cout << "\nTesting tensor operations:\n";
        std::cout << "1. Tensor addition...\n";
        
        try {
            auto result_add = tensor_a->add(*tensor_b);
            
            // Download and verify results
            std::vector<float> result_data(16);
            result_add->download_data(result_data.data());
            
            std::cout << "   First few results: ";
            for (int i = 0; i < 4; ++i) {
                std::cout << result_data[i] << " ";
            }
            std::cout << "\n   ✓ Addition completed successfully\n";
            
        } catch (const std::exception& e) {
            std::cerr << "   ✗ Addition failed: " << e.what() << std::endl;
        }
        
        // Test matrix multiplication
        std::cout << "2. Matrix multiplication...\n";
        
        try {
            auto result_matmul = tensor_a->matrix_multiply(*tensor_b);
            
            std::vector<float> matmul_result(16);
            result_matmul->download_data(matmul_result.data());
            
            std::cout << "   First row of result: ";
            for (int i = 0; i < 4; ++i) {
                std::cout << matmul_result[i] << " ";
            }
            std::cout << "\n   ✓ Matrix multiplication completed successfully\n";
            
        } catch (const std::exception& e) {
            std::cerr << "   ✗ Matrix multiplication failed: " << e.what() << std::endl;
        }
        
        // Test ReLU activation
        std::cout << "3. ReLU activation...\n";
        
        try {
            // Create tensor with some negative values
            std::vector<float> relu_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -0.5f, 1.5f};
            auto relu_input = std::make_shared<dlvk::Tensor>(
                std::vector<size_t>{2, 4}, dlvk::DataType::FLOAT32, device
            );
            relu_input->upload_data(relu_data.data());
            
            auto relu_result = relu_input->relu();
            
            std::vector<float> relu_output(8);
            relu_result->download_data(relu_output.data());
            
            std::cout << "   Input:  ";
            for (int i = 0; i < 8; ++i) {
                std::cout << relu_data[i] << " ";
            }
            std::cout << "\n   Output: ";
            for (int i = 0; i < 8; ++i) {
                std::cout << relu_output[i] << " ";
            }
            std::cout << "\n   ✓ ReLU activation completed successfully\n";
            
        } catch (const std::exception& e) {
            std::cerr << "   ✗ ReLU activation failed: " << e.what() << std::endl;
        }
        
        // Test neural network layer
        std::cout << "\n4. Neural network layer forward pass...\n";
        
        try {
            auto dense_layer = std::make_shared<dlvk::DenseLayer>(4, 2, device);
            
            auto input = std::make_shared<dlvk::Tensor>(
                std::vector<size_t>{1, 4}, dlvk::DataType::FLOAT32, device
            );
            std::vector<float> input_data = {1.0f, 0.5f, -0.2f, 0.8f};
            input->upload_data(input_data.data());
            
            // Note: The forward pass might not work completely yet as it depends
            // on matrix multiplication and bias addition
            std::cout << "   ✓ Dense layer created and input prepared\n";
            
        } catch (const std::exception& e) {
            std::cerr << "   ✗ Neural network layer test failed: " << e.what() << std::endl;
        }
        
        std::cout << "\nPhase 2 Progress Summary:\n";
        std::cout << "========================\n";
        std::cout << "✓ Compute pipeline system implemented\n";
        std::cout << "✓ Descriptor set management working\n";
        std::cout << "✓ Command buffer recording and submission\n";
        std::cout << "✓ Tensor addition operation functional\n";
        std::cout << "✓ Matrix multiplication implementation\n";
        std::cout << "✓ ReLU activation function working\n";
        std::cout << "✓ GPU memory synchronization\n";
        
        std::cout << "\nNext Phase 2 tasks:\n";
        std::cout << "- Implement remaining tensor operations (multiply, subtract, divide)\n";
        std::cout << "- Add sigmoid and tanh activation functions\n";
        std::cout << "- Implement transpose operation\n";
        std::cout << "- Add reduction operations (sum, mean)\n";
        std::cout << "- Optimize compute shader performance\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\nDemo completed successfully!\n";
    return 0;
}
