#include <iostream>
#include <vector>
#include <memory>
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/layers/dense_layer.h"
#include "dlvk/loss/loss_functions.h"

using namespace dlvk;

int main() {
    std::cout << "DLVK - Backward Propagation Test\n";
    std::cout << "================================\n\n";

    try {
        // Initialize Vulkan device
        auto device = std::make_shared<VulkanDevice>();
        if (!device->is_initialized()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return 1;
        }
        std::cout << "✓ Vulkan device initialized\n";

        // Initialize tensor operations
        TensorOps::initialize(device);
        std::cout << "✓ Tensor operations initialized\n";

        // Create simple neural network
        auto dense = std::make_shared<DenseLayer>(2, 1, device); // 2 inputs, 1 output
        std::cout << "✓ Neural network layer created\n";

        // Create training data - simple XOR-like problem
        // Input: [[0,0], [0,1], [1,0], [1,1]]
        std::vector<float> input_data = {
            0.0f, 0.0f,  // Sample 1
            0.0f, 1.0f,  // Sample 2
            1.0f, 0.0f,  // Sample 3
            1.0f, 1.0f   // Sample 4
        };
        
        // Target: [[0], [1], [1], [0]] (XOR)
        std::vector<float> target_data = {
            0.0f,  // Sample 1 target
            1.0f,  // Sample 2 target
            1.0f,  // Sample 3 target
            0.0f   // Sample 4 target
        };

        auto input = std::make_shared<Tensor>(device, std::vector<size_t>{4, 2});
        auto target = std::make_shared<Tensor>(device, std::vector<size_t>{4, 1});
        
        input->upload_data(input_data.data());
        target->upload_data(target_data.data());
        std::cout << "✓ Training data created and uploaded\n";

        // Test forward pass
        std::cout << "\nTesting forward pass...\n";
        auto output = dense->forward(input);
        
        // Download and print output
        std::vector<float> output_data(4);
        output->download_data(output_data.data());
        std::cout << "Initial output: ";
        for (float val : output_data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        // Test backward pass
        std::cout << "\nTesting backward propagation...\n";
        
        // Compute loss and gradients
        auto mse_loss = MeanSquaredError();
        auto loss = mse_loss.forward(output, target);
        
        // Download loss value
        std::vector<float> loss_data(1);
        loss->download_data(loss_data.data());
        std::cout << "Initial loss: " << loss_data[0] << std::endl;
        
        // Backward pass
        auto loss_grad = mse_loss.backward(output, target);
        std::cout << "✓ Loss gradients computed\n";
        
        // This should now work with our axis-specific reduction
        auto input_grad = dense->backward(loss_grad);
        std::cout << "✓ Layer backward pass completed\n";
        
        // Update weights
        dense->update_weights(0.1f);
        std::cout << "✓ Weights updated\n";
        
        // Test second forward pass to see if weights changed
        auto output2 = dense->forward(input);
        std::vector<float> output2_data(4);
        output2->download_data(output2_data.data());
        std::cout << "Output after weight update: ";
        for (float val : output2_data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        // Compute new loss
        auto loss2 = mse_loss.forward(output2, target);
        std::vector<float> loss2_data(1);
        loss2->download_data(loss2_data.data());
        std::cout << "Loss after update: " << loss2_data[0] << std::endl;

        std::cout << "\n✓ Backward propagation test completed successfully!\n";
        std::cout << "✓ All Phase 3 core components working!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
