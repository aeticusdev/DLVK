#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <functional>

#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/loss/loss_functions.h"

using namespace dlvk;

int main() {
    std::cout << "=== GPU Performance Validation Test ===" << std::endl;
    
    // Initialize
    auto device = std::make_shared<VulkanDevice>();
    device->initialize();
    auto tensor_ops = std::make_shared<TensorOps>(device);
    tensor_ops->initialize();
    Tensor::set_tensor_ops(tensor_ops);
    
    // Test data
    size_t batch_size = 2048;  // Larger for better GPU utilization  
    size_t num_classes = 512;
    std::vector<size_t> shape = {batch_size, num_classes};
    
    auto predictions = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    auto targets = std::make_shared<Tensor>(shape, DataType::FLOAT32, device);
    
    // Initialize data
    std::vector<float> pred_data(batch_size * num_classes);
    std::vector<float> target_data(batch_size * num_classes);
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            size_t idx = i * num_classes + j;
            pred_data[idx] = 0.1f + (j % 10) * 0.08f;
            target_data[idx] = (j == (i % num_classes)) ? 1.0f : 0.0f;
        }
    }
    
    predictions->upload_data(pred_data.data());
    targets->upload_data(target_data.data());
    
    std::cout << "✓ Data: [" << batch_size << ", " << num_classes << "] = " 
              << (batch_size * num_classes) << " elements" << std::endl;
    
    // Test CrossEntropy multiple times for consistency
    CrossEntropyLoss ce_loss;
    
    std::cout << "\n--- CrossEntropy Performance Test ---" << std::endl;
    
    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto loss = ce_loss.forward(predictions, targets);
        auto gradient = ce_loss.backward(predictions, targets);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        auto throughput = (double)(batch_size * num_classes) / (total_time / 1000000.0) / 1000000.0;
        
        std::cout << "Run " << (i+1) << ": " << total_time << " μs, " 
                  << throughput << " million elem/sec";
        
        if (total_time < 5000) std::cout << " 🚀";
        else std::cout << " ⚠️";
        std::cout << std::endl;
    }
    
    // Test BinaryCrossEntropy
    BinaryCrossEntropyLoss bce_loss;
    
    std::cout << "\n--- BinaryCrossEntropy Performance Test ---" << std::endl;
    
    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto loss = bce_loss.forward(predictions, targets);
        auto gradient = bce_loss.backward(predictions, targets);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        auto throughput = (double)(batch_size * num_classes) / (total_time / 1000000.0) / 1000000.0;
        
        std::cout << "Run " << (i+1) << ": " << total_time << " μs, " 
                  << throughput << " million elem/sec";
        
        if (total_time < 5000) std::cout << " 🚀";
        else std::cout << " ⚠️";
        std::cout << std::endl;
    }
    
    std::cout << "\n🎯 GPU acceleration validation complete!" << std::endl;
    return 0;
}
