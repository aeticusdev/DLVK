#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>

#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/loss/loss_functions.h"

using namespace dlvk;

int main() {
    std::cout << "=== BinaryCrossEntropy GPU Performance Test ===" << std::endl;
    
    // Initialize Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return -1;
    }
    
    std::cout << "✓ GPU: " << device->get_device_name() << std::endl;
    
    // Create large test data for meaningful performance measurement
    size_t batch_size = 1024;  // Large batch for GPU utilization
    size_t output_size = 1024;  // Large output size for better measurement
    std::vector<size_t> shape = {batch_size, output_size};
    
    auto predictions = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    auto targets = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    
    // Initialize with binary classification test data
    std::vector<float> pred_data(batch_size * output_size);
    std::vector<float> target_data(batch_size * output_size);
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        // Predictions between 0.1 and 0.9 to avoid log(0) issues
        pred_data[i] = 0.1f + (i % 100) * 0.008f;  // Values from 0.1 to 0.892
        target_data[i] = (i % 7 < 3) ? 1.0f : 0.0f;  // Binary targets
    }
    
    predictions->upload_data(pred_data.data());
    targets->upload_data(target_data.data());
    
    size_t total_elements = batch_size * output_size;
    std::cout << "✓ Test data: [" << batch_size << ", " << output_size << "] = " 
              << total_elements << " elements" << std::endl;
    
    // Test BinaryCrossEntropy Loss
    BinaryCrossEntropyLoss bce_loss;
    
    std::cout << "\n--- GPU Performance Test ---" << std::endl;
    
    // Warmup/First run (includes initialization)
    auto start = std::chrono::high_resolution_clock::now();
    auto loss1 = bce_loss.forward(predictions, targets);
    auto end = std::chrono::high_resolution_clock::now();
    auto init_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float loss_value;
    loss1->download_data(&loss_value);
    std::cout << "✓ First run (with init): " << init_time << " μs, Loss: " << loss_value << std::endl;
    
    // Second run (should be faster - cached pipelines)
    start = std::chrono::high_resolution_clock::now();
    auto loss2 = bce_loss.forward(predictions, targets);
    end = std::chrono::high_resolution_clock::now();
    auto fast_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "✓ Second run (cached): " << fast_time << " μs" << std::endl;
    
    // Multiple runs for better average
    const int num_runs = 50;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        bce_loss.forward(predictions, targets);
    }
    end = std::chrono::high_resolution_clock::now();
    auto multi_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto avg_time = multi_time / num_runs;
    
    std::cout << "✓ " << num_runs << " runs: " << multi_time << " μs (avg: " << avg_time << " μs)" << std::endl;
    
    // Test backward pass
    start = std::chrono::high_resolution_clock::now();
    auto gradient = bce_loss.backward(predictions, targets);
    end = std::chrono::high_resolution_clock::now();
    auto backward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "✓ Backward pass: " << backward_time << " μs" << std::endl;
    
    // Calculate throughput
    double forward_throughput = (double)total_elements / (avg_time / 1000000.0) / 1000000.0;  // Million elements/sec
    double backward_throughput = (double)total_elements / (backward_time / 1000000.0) / 1000000.0;
    
    std::cout << "\n--- Performance Summary ---" << std::endl;
    std::cout << "Data size: " << total_elements << " elements" << std::endl;
    std::cout << "Forward (avg): " << avg_time << " μs" << std::endl;
    std::cout << "Backward: " << backward_time << " μs" << std::endl;
    std::cout << "Forward Throughput: " << std::fixed << std::setprecision(2) << forward_throughput << " million elements/second" << std::endl;
    std::cout << "Backward Throughput: " << std::fixed << std::setprecision(2) << backward_throughput << " million elements/second" << std::endl;
    
    // Performance assessment
    if (forward_throughput > 200.0) {
        std::cout << "🚀 EXCELLENT PERFORMANCE! Optimized GPU shaders working." << std::endl;
    } else if (forward_throughput > 100.0) {
        std::cout << "✅ GOOD PERFORMANCE! GPU acceleration confirmed." << std::endl;
    } else if (forward_throughput > 50.0) {
        std::cout << "⚠️  MODERATE PERFORMANCE. GPU working but room for optimization." << std::endl;
    } else {
        std::cout << "❌ LOW PERFORMANCE. Potential CPU fallback or inefficient GPU usage." << std::endl;
    }
    
    // Compare with previous benchmark
    if (forward_throughput > 64.0) {
        double improvement = forward_throughput / 64.0;
        std::cout << "📈 Improvement over previous: " << std::fixed << std::setprecision(1) << improvement << "x faster!" << std::endl;
    }
    
    std::cout << "\n=== BinaryCrossEntropy Performance Test Completed! ===" << std::endl;
    
    return 0;
}
