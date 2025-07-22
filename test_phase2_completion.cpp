#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <vector>
#include <iomanip>

void test_element_wise_operations(std::shared_ptr<dlvk::VulkanDevice> device) {
    std::cout << "\n=== Testing Element-wise Operations ===\n";
    
    auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
    if (!tensor_ops->initialize()) {
        std::cout << "✗ Failed to initialize tensor operations" << std::endl;
        return;
    }
    dlvk::Tensor::set_tensor_ops(tensor_ops);
    
    // Create test tensors
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {2.0f, 1.0f, 2.0f, 1.0f};
    
    auto tensor_a = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device
    );
    auto tensor_b = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device
    );
    
    tensor_a->upload_data(data_a.data());
    tensor_b->upload_data(data_b.data());
    
    // Test addition
    try {
        auto result_add = tensor_a->add(*tensor_b);
        std::vector<float> output(4);
        result_add->download_data(output.data());
        std::cout << "✓ Addition: ";
        for (float val : output) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Addition failed: " << e.what() << std::endl;
    }
    
    // Test multiplication
    try {
        auto result_mul = tensor_a->multiply(*tensor_b);
        std::vector<float> output(4);
        result_mul->download_data(output.data());
        std::cout << "✓ Multiplication: ";
        for (float val : output) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Multiplication failed: " << e.what() << std::endl;
    }
    
    // Test subtraction
    try {
        auto result_sub = tensor_a->subtract(*tensor_b);
        std::vector<float> output(4);
        result_sub->download_data(output.data());
        std::cout << "✓ Subtraction: ";
        for (float val : output) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Subtraction failed: " << e.what() << std::endl;
    }
    
    // Test division
    try {
        auto result_div = tensor_a->divide(*tensor_b);
        std::vector<float> output(4);
        result_div->download_data(output.data());
        std::cout << "✓ Division: ";
        for (float val : output) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Division failed: " << e.what() << std::endl;
    }
}

void test_activation_functions(std::shared_ptr<dlvk::VulkanDevice> device) {
    std::cout << "\n=== Testing Activation Functions ===\n";
    
    auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
    if (!tensor_ops->initialize()) {
        std::cout << "✗ Failed to initialize tensor operations" << std::endl;
        return;
    }
    dlvk::Tensor::set_tensor_ops(tensor_ops);
    
    std::vector<float> test_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto input_tensor = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{5}, dlvk::DataType::FLOAT32, device
    );
    input_tensor->upload_data(test_data.data());
    
    // Test ReLU
    try {
        auto result = input_tensor->relu();
        std::vector<float> output(5);
        result->download_data(output.data());
        std::cout << "✓ ReLU: ";
        for (float val : output) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ ReLU failed: " << e.what() << std::endl;
    }
    
    // Test Sigmoid
    try {
        auto result = input_tensor->sigmoid();
        std::vector<float> output(5);
        result->download_data(output.data());
        std::cout << "✓ Sigmoid: ";
        for (float val : output) std::cout << std::fixed << std::setprecision(3) << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Sigmoid failed: " << e.what() << std::endl;
    }
    
    // Test Tanh
    try {
        auto result = input_tensor->tanh();
        std::vector<float> output(5);
        result->download_data(output.data());
        std::cout << "✓ Tanh: ";
        for (float val : output) std::cout << std::fixed << std::setprecision(3) << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Tanh failed: " << e.what() << std::endl;
    }
    
    // Test Softmax
    try {
        std::vector<float> softmax_data = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
        auto softmax_input = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{2, 3}, dlvk::DataType::FLOAT32, device
        );
        softmax_input->upload_data(softmax_data.data());
        
        auto result = softmax_input->softmax();
        std::vector<float> output(6);
        result->download_data(output.data());
        std::cout << "✓ Softmax: ";
        for (float val : output) std::cout << std::fixed << std::setprecision(3) << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Softmax failed: " << e.what() << std::endl;
    }
}

void test_matrix_operations(std::shared_ptr<dlvk::VulkanDevice> device) {
    std::cout << "\n=== Testing Matrix Operations ===\n";
    
    auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
    if (!tensor_ops->initialize()) {
        std::cout << "✗ Failed to initialize tensor operations" << std::endl;
        return;
    }
    dlvk::Tensor::set_tensor_ops(tensor_ops);
    
    // Test Matrix Multiplication
    try {
        std::vector<float> data_a = {1, 2, 3, 4, 5, 6};  // 2x3
        std::vector<float> data_b = {1, 2, 3, 4, 5, 6};  // 3x2
        
        auto tensor_a = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{2, 3}, dlvk::DataType::FLOAT32, device
        );
        auto tensor_b = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{3, 2}, dlvk::DataType::FLOAT32, device
        );
        
        tensor_a->upload_data(data_a.data());
        tensor_b->upload_data(data_b.data());
        
        auto result = tensor_a->matrix_multiply(*tensor_b);
        std::vector<float> output(4);  // 2x2 result
        result->download_data(output.data());
        std::cout << "✓ Matrix Multiply: ";
        for (float val : output) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Matrix Multiply failed: " << e.what() << std::endl;
    }
    
    // Test Transpose
    try {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};  // 2x3
        auto input_tensor = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{2, 3}, dlvk::DataType::FLOAT32, device
        );
        input_tensor->upload_data(data.data());
        
        auto result = input_tensor->transpose();
        std::vector<float> output(6);  // 3x2 result
        result->download_data(output.data());
        std::cout << "✓ Transpose: ";
        for (float val : output) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Transpose failed: " << e.what() << std::endl;
    }
}

void test_reduction_operations(std::shared_ptr<dlvk::VulkanDevice> device) {
    std::cout << "\n=== Testing Reduction Operations ===\n";
    
    auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
    if (!tensor_ops->initialize()) {
        std::cout << "✗ Failed to initialize tensor operations" << std::endl;
        return;
    }
    dlvk::Tensor::set_tensor_ops(tensor_ops);
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input_tensor = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device
    );
    input_tensor->upload_data(data.data());
    
    // Test Sum
    try {
        auto result = input_tensor->sum();
        std::vector<float> output(1);
        result->download_data(output.data());
        std::cout << "✓ Sum: " << output[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Sum failed: " << e.what() << std::endl;
    }
    
    // Test Mean
    try {
        auto result = input_tensor->mean();
        std::vector<float> output(1);
        result->download_data(output.data());
        std::cout << "✓ Mean: " << output[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Mean failed: " << e.what() << std::endl;
    }
    
    // Test Max
    try {
        auto result = input_tensor->max();
        std::vector<float> output(1);
        result->download_data(output.data());
        std::cout << "✓ Max: " << output[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Max failed: " << e.what() << std::endl;
    }
    
    // Test Min
    try {
        auto result = input_tensor->min();
        std::vector<float> output(1);
        result->download_data(output.data());
        std::cout << "✓ Min: " << output[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Min failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "DLVK Phase 2 Completion Test\n";
    std::cout << "============================\n";
    
    try {
        // Initialize Vulkan device
        auto device = std::make_shared<dlvk::VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return -1;
        }
        std::cout << "✓ Vulkan device initialized\n";
        
        test_element_wise_operations(device);
        test_activation_functions(device);
        test_matrix_operations(device);
        test_reduction_operations(device);
        
        std::cout << "\n=== Phase 2 Completion Summary ===\n";
        std::cout << "Review the output above to see which operations are working.\n";
        std::cout << "Any operation marked with ✗ needs implementation or fixing.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
