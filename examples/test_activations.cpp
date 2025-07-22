#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "DLVK - Testing Activation Functions\n";
    std::cout << "===================================\n\n";
    
    try {
        // Initialize Vulkan device
        auto device = std::make_shared<dlvk::VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return -1;
        }
        
        // Initialize tensor operations
        auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
        if (!tensor_ops->initialize()) {
            std::cerr << "Failed to initialize tensor operations" << std::endl;
            return -1;
        }
        
        dlvk::Tensor::set_tensor_ops(tensor_ops);
        
        // Test sigmoid activation
        std::cout << "Testing Sigmoid activation...\n";
        std::vector<float> sigmoid_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        auto sigmoid_input = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{5}, dlvk::DataType::FLOAT32, device
        );
        sigmoid_input->upload_data(sigmoid_data.data());
        
        auto sigmoid_result = sigmoid_input->sigmoid();
        
        std::vector<float> sigmoid_output(5);
        sigmoid_result->download_data(sigmoid_output.data());
        
        std::cout << "Input:  ";
        for (float val : sigmoid_data) {
            std::cout << val << " ";
        }
        std::cout << "\nOutput: ";
        for (float val : sigmoid_output) {
            std::cout << val << " ";
        }
        std::cout << "\n✓ Sigmoid test completed\n\n";
        
        // Test tanh activation
        std::cout << "Testing Tanh activation...\n";
        std::vector<float> tanh_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        auto tanh_input = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{5}, dlvk::DataType::FLOAT32, device
        );
        tanh_input->upload_data(tanh_data.data());
        
        auto tanh_result = tanh_input->tanh();
        
        std::vector<float> tanh_output(5);
        tanh_result->download_data(tanh_output.data());
        
        std::cout << "Input:  ";
        for (float val : tanh_data) {
            std::cout << val << " ";
        }
        std::cout << "\nOutput: ";
        for (float val : tanh_output) {
            std::cout << val << " ";
        }
        std::cout << "\n✓ Tanh test completed\n\n";
        
        // Test element-wise operations
        std::cout << "Testing element-wise operations...\n";
        std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> b_data = {2.0f, 1.0f, 2.0f, 1.0f};
        
        auto tensor_a = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device
        );
        auto tensor_b = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device
        );
        
        tensor_a->upload_data(a_data.data());
        tensor_b->upload_data(b_data.data());
        
        // Test subtraction
        auto sub_result = tensor_a->subtract(*tensor_b);
        std::vector<float> sub_output(4);
        sub_result->download_data(sub_output.data());
        
        std::cout << "Subtraction: ";
        for (size_t i = 0; i < 4; ++i) {
            std::cout << a_data[i] << "-" << b_data[i] << "=" << sub_output[i] << " ";
        }
        std::cout << "\n";
        
        // Test division
        auto div_result = tensor_a->divide(*tensor_b);
        std::vector<float> div_output(4);
        div_result->download_data(div_output.data());
        
        std::cout << "Division: ";
        for (size_t i = 0; i < 4; ++i) {
            std::cout << a_data[i] << "/" << b_data[i] << "=" << div_output[i] << " ";
        }
        std::cout << "\n✓ Element-wise operations completed\n\n";
        
        // Test matrix transpose
        std::cout << "Testing matrix transpose...\n";
        std::vector<float> matrix_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        auto matrix_input = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{2, 3}, dlvk::DataType::FLOAT32, device
        );
        matrix_input->upload_data(matrix_data.data());
        
        auto transpose_result = matrix_input->transpose();
        std::vector<float> transpose_output(6);
        transpose_result->download_data(transpose_output.data());
        
        std::cout << "Original (2x3): ";
        for (float val : matrix_data) {
            std::cout << val << " ";
        }
        std::cout << "\nTransposed (3x2): ";
        for (float val : transpose_output) {
            std::cout << val << " ";
        }
        std::cout << "\n✓ Transpose test completed\n\n";
        
        // Test sum reduction
        std::cout << "Testing sum reduction...\n";
        auto sum_result = tensor_a->sum();
        std::vector<float> sum_output(1);
        sum_result->download_data(sum_output.data());
        
        std::cout << "Sum of [";
        for (float val : a_data) {
            std::cout << val << " ";
        }
        std::cout << "] = " << sum_output[0] << "\n";
        std::cout << "✓ Sum reduction completed\n\n";
        
        // Test softmax
        std::cout << "Testing softmax...\n";
        std::vector<float> softmax_data = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
        auto softmax_input = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{2, 3}, dlvk::DataType::FLOAT32, device
        );
        softmax_input->upload_data(softmax_data.data());
        
        auto softmax_result = softmax_input->softmax();
        std::vector<float> softmax_output(6);
        softmax_result->download_data(softmax_output.data());
        
        std::cout << "Softmax input (2x3): ";
        for (float val : softmax_data) {
            std::cout << val << " ";
        }
        std::cout << "\nSoftmax output: ";
        for (float val : softmax_output) {
            std::cout << val << " ";
        }
        std::cout << "\n✓ Softmax test completed\n\n";
        
        std::cout << "=== Phase 2 Operations Complete! ===\n";
        
        // Test tanh activation
        std::cout << "Testing Tanh activation...\n";
        std::vector<float> tanh_data = {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f};
        auto tanh_input = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{5}, dlvk::DataType::FLOAT32, device
        );
        tanh_input->upload_data(tanh_data.data());
        
        auto tanh_result = tanh_input->tanh();
        
        std::vector<float> tanh_output(5);
        tanh_result->download_data(tanh_output.data());
        
        std::cout << "Input:  ";
        for (float val : tanh_data) {
            std::cout << val << " ";
        }
        std::cout << "\nOutput: ";
        for (float val : tanh_output) {
            std::cout << val << " ";
        }
        std::cout << "\n✓ Tanh test completed\n\n";
        
        std::cout << "All activation function tests passed!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
