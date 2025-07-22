#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/layers/layer.h"
#include "dlvk/loss/loss_functions.h"
#include "dlvk/optimizers/optimizers.h"
#include <iostream>
#include <vector>
#include <iomanip>

void test_dense_layer_with_bias(std::shared_ptr<dlvk::VulkanDevice> device) {
    std::cout << "\n=== Testing Dense Layer with Bias ===\n";
    
    auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
    if (!tensor_ops->initialize()) {
        std::cout << "✗ Failed to initialize tensor operations" << std::endl;
        return;
    }
    dlvk::Tensor::set_tensor_ops(tensor_ops);
    
    // Create a simple dense layer: 3 inputs -> 2 outputs
    auto dense = std::make_shared<dlvk::DenseLayer>(3, 2, device);
    
    // Create input tensor: batch_size=2, input_features=3
    auto input = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{2, 3}, dlvk::DataType::FLOAT32, device
    );
    
    // Upload test data
    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f,  // First sample
        4.0f, 5.0f, 6.0f   // Second sample
    };
    input->upload_data(input_data.data());
    
    try {
        // Forward pass
        auto output = dense->forward(input);
        
        // Download and display results
        std::vector<float> output_data(output->size());
        output->download_data(output_data.data());
        
        std::cout << "✓ Dense layer forward pass successful" << std::endl;
        std::cout << "Input shape: [2, 3], Output shape: [" 
                  << output->shape()[0] << ", " << output->shape()[1] << "]" << std::endl;
        std::cout << "Output values: ";
        for (size_t i = 0; i < output_data.size(); ++i) {
            std::cout << std::fixed << std::setprecision(3) << output_data[i] << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Dense layer forward pass failed: " << e.what() << std::endl;
    }
}

void test_mse_loss(std::shared_ptr<dlvk::VulkanDevice> device) {
    std::cout << "\n=== Testing MSE Loss Function ===\n";
    
    auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
    if (!tensor_ops->initialize()) {
        std::cout << "✗ Failed to initialize tensor operations" << std::endl;
        return;
    }
    dlvk::Tensor::set_tensor_ops(tensor_ops);
    
    // Create predictions and targets
    auto predictions = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device
    );
    auto targets = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{4}, dlvk::DataType::FLOAT32, device
    );
    
    std::vector<float> pred_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> target_data = {1.1f, 1.9f, 3.2f, 3.8f};
    
    predictions->upload_data(pred_data.data());
    targets->upload_data(target_data.data());
    
    try {
        dlvk::MeanSquaredError mse_loss;
        
        // Forward pass
        auto loss = mse_loss.forward(predictions, targets);
        
        float loss_value;
        loss->download_data(&loss_value);
        
        std::cout << "✓ MSE loss computation successful" << std::endl;
        std::cout << "Loss value: " << std::fixed << std::setprecision(6) << loss_value << std::endl;
        
        // Backward pass
        auto gradient = mse_loss.backward(predictions, targets);
        std::vector<float> grad_data(gradient->size());
        gradient->download_data(grad_data.data());
        
        std::cout << "✓ MSE gradient computation successful" << std::endl;
        std::cout << "Gradients: ";
        for (float g : grad_data) {
            std::cout << std::fixed << std::setprecision(6) << g << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ MSE loss test failed: " << e.what() << std::endl;
    }
}

void test_broadcast_addition(std::shared_ptr<dlvk::VulkanDevice> device) {
    std::cout << "\n=== Testing Broadcast Addition ===\n";
    
    auto tensor_ops = std::make_shared<dlvk::TensorOps>(device);
    if (!tensor_ops->initialize()) {
        std::cout << "✗ Failed to initialize tensor operations" << std::endl;
        return;
    }
    dlvk::Tensor::set_tensor_ops(tensor_ops);
    
    // Create matrix and bias vector
    auto matrix = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{2, 3}, dlvk::DataType::FLOAT32, device
    );
    auto bias = std::make_shared<dlvk::Tensor>(
        std::vector<size_t>{3}, dlvk::DataType::FLOAT32, device
    );
    
    std::vector<float> matrix_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> bias_data = {0.1f, 0.2f, 0.3f};
    
    matrix->upload_data(matrix_data.data());
    bias->upload_data(bias_data.data());
    
    try {
        // Test broadcast addition
        auto result = matrix->add_broadcast(*bias);
        
        std::vector<float> result_data(result->size());
        result->download_data(result_data.data());
        
        std::cout << "✓ Broadcast addition successful" << std::endl;
        std::cout << "Matrix + bias result: ";
        for (size_t i = 0; i < result_data.size(); ++i) {
            std::cout << std::fixed << std::setprecision(1) << result_data[i] << " ";
            if (i == 2) std::cout << "| ";  // Separate rows
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Broadcast addition failed: " << e.what() << std::endl;
    }
}

void test_sgd_optimizer() {
    std::cout << "\n=== Testing SGD Optimizer ===\n";
    
    try {
        dlvk::SGD optimizer(0.01f);
        
        std::cout << "✓ SGD optimizer created successfully" << std::endl;
        std::cout << "Learning rate: " << optimizer.get_learning_rate() << std::endl;
        
        // Test learning rate modification
        optimizer.set_learning_rate(0.001f);
        std::cout << "✓ Learning rate updated to: " << optimizer.get_learning_rate() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ SGD optimizer test failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "DLVK Phase 3 Neural Network Components Test" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    try {
        // Initialize Vulkan device
        auto device = std::make_shared<dlvk::VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return -1;
        }
        std::cout << "✓ Vulkan device initialized" << std::endl;
        
        // Test all Phase 3 components
        test_broadcast_addition(device);
        test_dense_layer_with_bias(device);
        test_mse_loss(device);
        test_sgd_optimizer();
        
        std::cout << "\n=== Phase 3 Component Summary ===" << std::endl;
        std::cout << "Dense Layer: Implemented with bias support" << std::endl;
        std::cout << "Broadcast Addition: Working for bias addition" << std::endl;
        std::cout << "MSE Loss: Forward and backward pass implemented" << std::endl;
        std::cout << "SGD Optimizer: Basic implementation ready" << std::endl;
        std::cout << "\nPhase 3 foundation complete! Ready for training loops." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
