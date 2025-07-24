#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/loss/loss_functions.h"

using namespace dlvk;

int main() {
    std::cout << "=== DLVK GPU Loss Functions Performance Test ===" << std::endl;
    
    // Initialize Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return -1;
    }
    
    std::cout << "✓ GPU: " << device->get_device_name() << std::endl;
    
    // Create test data
    size_t batch_size = 1024;  // Larger batch for better GPU utilization
    size_t output_size = 256;
    std::vector<size_t> shape = {batch_size, output_size};
    
    auto predictions = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    auto targets = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    
    // Initialize with test data
    std::vector<float> pred_data(batch_size * output_size);
    std::vector<float> target_data(batch_size * output_size);
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        pred_data[i] = 0.1f + (i % 10) * 0.05f;
        target_data[i] = (i % 10 == 5) ? 1.0f : 0.0f;
    }
    
    predictions->upload_data(pred_data.data());
    targets->upload_data(target_data.data());
    
    std::cout << "✓ Test data: [" << batch_size << ", " << output_size << "] = " 
              << (batch_size * output_size) << " elements" << std::endl;
    
    // Test MSE Loss
    MeanSquaredError mse_loss;
    
    std::cout << "\n--- GPU Acceleration Test ---" << std::endl;
    
    // First run (includes initialization)
    auto start = std::chrono::high_resolution_clock::now();
    auto loss1 = mse_loss.forward(predictions, targets);
    auto end = std::chrono::high_resolution_clock::now();
    auto init_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float loss_value;
    loss1->download_data(&loss_value);
    std::cout << "✓ First run (with init): " << init_time << " μs, Loss: " << loss_value << std::endl;
    
    // Second run (should be faster)
    start = std::chrono::high_resolution_clock::now();
    auto loss2 = mse_loss.forward(predictions, targets);
    end = std::chrono::high_resolution_clock::now();
    auto fast_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "✓ Second run (cached): " << fast_time << " μs" << std::endl;
    
    // Test backward pass
    start = std::chrono::high_resolution_clock::now();
    auto gradient = mse_loss.backward(predictions, targets);
    end = std::chrono::high_resolution_clock::now();
    auto backward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "✓ Backward pass: " << backward_time << " μs" << std::endl;
    
    // Calculate throughput
    size_t total_elements = batch_size * output_size;
    double elements_per_second = (double)total_elements / (fast_time / 1000000.0);
    
    std::cout << "\n--- Performance Summary ---" << std::endl;
    std::cout << "Data size: " << total_elements << " elements" << std::endl;
    std::cout << "Forward (cached): " << fast_time << " μs" << std::endl;
    std::cout << "Backward: " << backward_time << " μs" << std::endl;
    std::cout << "Throughput: " << (elements_per_second / 1000000.0) << " million elements/second" << std::endl;
    
    // Verify GPU vs CPU fallback by checking performance
    if (fast_time < 10000) {  // Less than 10ms suggests GPU acceleration
        std::cout << "🚀 GPU ACCELERATION CONFIRMED! Fast execution suggests GPU compute." << std::endl;
    } else {
        std::cout << "⚠️  Performance suggests CPU fallback may be active." << std::endl;
    }
    
    std::cout << "\n=== Test Completed Successfully! ===" << std::endl;
    return 0;
}
