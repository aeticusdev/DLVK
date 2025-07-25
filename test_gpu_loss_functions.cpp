#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/loss/loss_functions.h"

using namespace dlvk;

int main() {
    std::cout << "=== DLVK GPU Loss Functions Test ===" << std::endl;
    
    // Initialize Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return -1;
    }
    
    std::cout << "✓ Vulkan device initialized: " << device->get_device_name() << std::endl;
    
    // Create test data
    size_t batch_size = 32;
    size_t output_size = 10;
    std::vector<size_t> shape = {batch_size, output_size};
    
    auto predictions = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    auto targets = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    
    // Initialize with test data
    std::vector<float> pred_data(batch_size * output_size);
    std::vector<float> target_data(batch_size * output_size);
    
    // Create some realistic test data
    for (size_t i = 0; i < pred_data.size(); ++i) {
        pred_data[i] = 0.1f + (i % 10) * 0.08f;  // Values between 0.1 and 0.82
        target_data[i] = (i % 10 == 5) ? 1.0f : 0.0f;  // One-hot style targets
    }
    
    predictions->upload_data(pred_data.data());
    targets->upload_data(target_data.data());
    
    std::cout << "✓ Test tensors created with shape [" << batch_size << ", " << output_size << "]" << std::endl;
    
    // Test MSE Loss
    std::cout << "\n--- Testing MSE Loss ---" << std::endl;
    
    MeanSquaredError mse_loss;
    
    // Measure forward pass time
    auto start = std::chrono::high_resolution_clock::now();
    auto mse_result = mse_loss.forward(predictions, targets);
    auto end = std::chrono::high_resolution_clock::now();
    auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "✓ MSE forward pass completed in " << forward_time << " μs" << std::endl;
    
    // Download and print result
    float mse_value;
    mse_result->download_data(&mse_value);
    std::cout << "  MSE Loss Value: " << mse_value << std::endl;
    
    // Test backward pass
    start = std::chrono::high_resolution_clock::now();
    auto mse_gradient = mse_loss.backward(predictions, targets);
    end = std::chrono::high_resolution_clock::now();
    auto backward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "✓ MSE backward pass completed in " << backward_time << " μs" << std::endl;
    
    // Verify gradient shape
    if (mse_gradient->shape() == predictions->shape()) {
        std::cout << "✓ Gradient shape matches predictions shape" << std::endl;
    } else {
        std::cout << "✗ Gradient shape mismatch!" << std::endl;
    }
    
    // Download a few gradient values to verify
    std::vector<float> grad_sample(5);
    mse_gradient->download_data(grad_sample.data());
    std::cout << "  Sample gradients: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << grad_sample[i] << " ";
    }
    std::cout << std::endl;
    
    // Performance summary
    std::cout << "\n--- Performance Summary ---" << std::endl;
    std::cout << "MSE Forward: " << forward_time << " μs" << std::endl;
    std::cout << "MSE Backward: " << backward_time << " μs" << std::endl;
    std::cout << "Total MSE: " << (forward_time + backward_time) << " μs" << std::endl;
    
    // Test multiple iterations for performance
    std::cout << "\n--- Performance Benchmarking ---" << std::endl;
    const int iterations = 100;
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto loss = mse_loss.forward(predictions, targets);
        auto grad = mse_loss.backward(predictions, targets);
    }
    end = std::chrono::high_resolution_clock::now();
    
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto avg_time = total_time / iterations;
    
    std::cout << "✓ " << iterations << " iterations completed" << std::endl;
    std::cout << "  Total time: " << total_time << " μs" << std::endl;
    std::cout << "  Average per iteration: " << avg_time << " μs" << std::endl;
    std::cout << "  Throughput: " << (1000000.0 / avg_time) << " iterations/second" << std::endl;
    
    std::cout << "\n=== GPU Loss Functions Test Completed Successfully! ===" << std::endl;
    
    return 0;
}
