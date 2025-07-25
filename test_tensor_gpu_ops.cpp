#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>

#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"

using namespace dlvk;
using namespace std::chrono;

int main() {
    std::cout << "=== DLVK GPU Tensor Operations Test ===" << std::endl;
    
    // Initialize Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return -1;
    }
    
    std::cout << "✓ GPU: " << device->get_device_name() << std::endl;
    
    // Initialize TensorOps
    auto tensor_ops = std::make_shared<TensorOps>(device);
    if (!tensor_ops->initialize()) {
        std::cerr << "Failed to initialize TensorOps" << std::endl;
        return -1;
    }
    
    // Set global TensorOps instance
    Tensor::set_tensor_ops(tensor_ops);
    
    std::cout << "✓ TensorOps initialized" << std::endl;
    
    // Create test tensors
    size_t test_size = 1024 * 1024; // 1M elements for meaningful performance test
    std::vector<size_t> shape = {1024, 1024};
    
    auto tensor_a = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    auto tensor_b = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    auto result = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    
    // Initialize test data
    std::vector<float> data_a(test_size, 2.5f);
    std::vector<float> data_b(test_size, 1.5f);
    
    tensor_a->upload_data(data_a.data());
    tensor_b->upload_data(data_b.data());
    
    std::cout << "✓ Test tensors created: " << test_size << " elements each" << std::endl;
    
    // === Test 1: Scalar Multiply ===
    std::cout << "\n--- Testing Scalar Multiply (GPU) ---" << std::endl;
    
    auto start = high_resolution_clock::now();
    if (tensor_ops->scalar_multiply(*tensor_a, 3.0f, *result)) {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        // Verify result
        std::vector<float> result_data(test_size);
        result->download_data(result_data.data());
        
        bool correct = (std::abs(result_data[0] - 7.5f) < 1e-6f);  // 2.5 * 3.0 = 7.5
        
        std::cout << "✓ Scalar multiply: " << duration.count() << " μs";
        std::cout << (correct ? " (CORRECT)" : " (INCORRECT)") << std::endl;
        
        double throughput = (double)test_size / (duration.count() / 1000000.0) / 1000000.0;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << throughput << " million elements/second" << std::endl;
    } else {
        std::cout << "❌ Scalar multiply failed - using CPU fallback" << std::endl;
        
        // Test CPU fallback
        auto cpu_result = tensor_a->multiply_scalar(3.0f);
        std::vector<float> cpu_data(test_size);
        cpu_result->download_data(cpu_data.data());
        std::cout << "  CPU fallback result: " << cpu_data[0] << " (expected 7.5)" << std::endl;
    }
    
    // === Test 2: Broadcast Add ===
    std::cout << "\n--- Testing Broadcast Add (GPU) ---" << std::endl;
    
    // Create bias vector
    auto bias = std::make_shared<Tensor>(std::vector<size_t>{1024}, DataType::FLOAT32, device);
    std::vector<float> bias_data(1024, 0.5f);
    bias->upload_data(bias_data.data());
    
    start = high_resolution_clock::now();
    if (tensor_ops->add_broadcast(*tensor_a, *bias, *result)) {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        // Verify result
        std::vector<float> result_data(test_size);
        result->download_data(result_data.data());
        
        bool correct = (std::abs(result_data[0] - 3.0f) < 1e-6f);  // 2.5 + 0.5 = 3.0
        
        std::cout << "✓ Broadcast add: " << duration.count() << " μs";
        std::cout << (correct ? " (CORRECT)" : " (INCORRECT)") << std::endl;
        
        double throughput = (double)test_size / (duration.count() / 1000000.0) / 1000000.0;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << throughput << " million elements/second" << std::endl;
    } else {
        std::cout << "❌ Broadcast add failed - falling back to CPU" << std::endl;
    }
    
    // === Test 3: Element-wise Sqrt ===
    std::cout << "\n--- Testing Element-wise Sqrt (GPU) ---" << std::endl;
    
    // Create sqrt test data
    std::vector<float> sqrt_data(test_size, 4.0f);
    tensor_a->upload_data(sqrt_data.data());
    
    start = high_resolution_clock::now();
    if (tensor_ops->element_wise_sqrt(*tensor_a, *result)) {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        // Verify result
        std::vector<float> result_data(test_size);
        result->download_data(result_data.data());
        
        bool correct = (std::abs(result_data[0] - 2.0f) < 1e-6f);  // sqrt(4.0) = 2.0
        
        std::cout << "✓ Element-wise sqrt: " << duration.count() << " μs";
        std::cout << (correct ? " (CORRECT)" : " (INCORRECT)") << std::endl;
        
        double throughput = (double)test_size / (duration.count() / 1000000.0) / 1000000.0;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << throughput << " million elements/second" << std::endl;
    } else {
        std::cout << "❌ Element-wise sqrt failed - using CPU fallback" << std::endl;
    }
    
    // === Test 4: Clamp ===
    std::cout << "\n--- Testing Clamp (GPU) ---" << std::endl;
    
    // Create clamp test data with values outside range
    std::vector<float> clamp_data(test_size);
    for (size_t i = 0; i < test_size; ++i) {
        clamp_data[i] = -5.0f + (i % 20) * 0.5f;  // Values from -5.0 to 5.0
    }
    tensor_a->upload_data(clamp_data.data());
    
    start = high_resolution_clock::now();
    if (tensor_ops->clamp(*tensor_a, -2.0f, 3.0f, *result)) {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        // Verify result
        std::vector<float> result_data(test_size);
        result->download_data(result_data.data());
        
        bool correct = (result_data[0] >= -2.0f && result_data[0] <= 3.0f);
        
        std::cout << "✓ Clamp: " << duration.count() << " μs";
        std::cout << (correct ? " (CORRECT)" : " (INCORRECT)") << std::endl;
        std::cout << "  Sample: " << clamp_data[0] << " -> " << result_data[0] << " (clamped to [-2.0, 3.0])" << std::endl;
        
        double throughput = (double)test_size / (duration.count() / 1000000.0) / 1000000.0;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << throughput << " million elements/second" << std::endl;
    } else {
        std::cout << "❌ Clamp failed - using CPU fallback" << std::endl;
    }
    
    std::cout << "\n=== GPU Tensor Operations Test Completed! ===" << std::endl;
    
    return 0;
}
