#include "dlvk/core/vulkan_device.h"
#include "dlvk/core/memory_pool_manager.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace dlvk;

void test_memory_pool_fast() {
    std::cout << "\n🚀 DLVK Phase 7.2 Fast Memory Pool Test" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Initialize Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return;
    }
    
    std::cout << "✓ Vulkan device initialized: " << device->get_device_name() << std::endl;
    
    // Initialize TensorOps once
    TensorOps tensor_ops(device);
    if (!tensor_ops.initialize()) {
        std::cerr << "Failed to initialize TensorOps" << std::endl;
        return;
    }
    std::cout << "✓ TensorOps initialized" << std::endl;
    
    // Test with smaller tensors for faster execution
    const size_t tensor_size = 128; // 128x128 instead of 512x512
    const int num_tensors = 10;     // 10 instead of 20
    const int num_ops = 5;          // 5 instead of 10
    
    std::cout << "\n📊 Test Configuration:" << std::endl;
    std::cout << "   Tensor size: " << tensor_size << "x" << tensor_size << " (" << (tensor_size * tensor_size * 4 / 1024) << "KB each)" << std::endl;
    std::cout << "   Number of tensors: " << num_tensors << std::endl;
    std::cout << "   Number of operations: " << num_ops << std::endl;
    
    // Test 1: Without memory pools
    std::cout << "\n🔵 Test 1: Direct Allocation (Baseline)" << std::endl;
    device->enable_memory_pools(false);
    
    auto start_direct = std::chrono::high_resolution_clock::now();
    
    std::vector<Tensor> input_tensors_direct;
    std::vector<Tensor> output_tensors_direct;
    
    for (int i = 0; i < num_tensors; ++i) {
        input_tensors_direct.emplace_back(std::vector<size_t>{tensor_size, tensor_size}, DataType::FLOAT32, device);
        output_tensors_direct.emplace_back(std::vector<size_t>{tensor_size, tensor_size}, DataType::FLOAT32, device);
        
        // Fill with test data
        std::vector<float> data(tensor_size * tensor_size, 1.0f + i * 0.1f);
        input_tensors_direct.back().upload_data(data.data());
    }
    
    auto alloc_end_direct = std::chrono::high_resolution_clock::now();
    
    // Perform compute operations
    for (int i = 0; i < num_ops; ++i) {
        tensor_ops.matrix_multiply(input_tensors_direct[i], input_tensors_direct[i + 1], output_tensors_direct[i]);
        tensor_ops.add(output_tensors_direct[i], input_tensors_direct[i + 2], output_tensors_direct[i]);
    }
    
    vkDeviceWaitIdle(device->get_device());
    auto end_direct = std::chrono::high_resolution_clock::now();
    
    auto alloc_direct = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end_direct - start_direct);
    auto total_direct = std::chrono::duration_cast<std::chrono::microseconds>(end_direct - start_direct);
    
    std::cout << "   Allocation time: " << alloc_direct.count() << " μs" << std::endl;
    std::cout << "   Total time: " << total_direct.count() << " μs" << std::endl;
    
    // Clear tensors
    input_tensors_direct.clear();
    output_tensors_direct.clear();
    
    // Test 2: With memory pools
    std::cout << "\n🟢 Test 2: Memory Pool Allocation (Optimized)" << std::endl;
    device->enable_memory_pools(true);
    
    auto start_pool = std::chrono::high_resolution_clock::now();
    
    std::vector<Tensor> input_tensors_pool;
    std::vector<Tensor> output_tensors_pool;
    
    for (int i = 0; i < num_tensors; ++i) {
        input_tensors_pool.emplace_back(std::vector<size_t>{tensor_size, tensor_size}, DataType::FLOAT32, device);
        output_tensors_pool.emplace_back(std::vector<size_t>{tensor_size, tensor_size}, DataType::FLOAT32, device);
        
        // Fill with test data
        std::vector<float> data(tensor_size * tensor_size, 1.0f + i * 0.1f);
        input_tensors_pool.back().upload_data(data.data());
    }
    
    auto alloc_end_pool = std::chrono::high_resolution_clock::now();
    
    // Perform same compute operations
    for (int i = 0; i < num_ops; ++i) {
        tensor_ops.matrix_multiply(input_tensors_pool[i], input_tensors_pool[i + 1], output_tensors_pool[i]);
        tensor_ops.add(output_tensors_pool[i], input_tensors_pool[i + 2], output_tensors_pool[i]);
    }
    
    vkDeviceWaitIdle(device->get_device());
    auto end_pool = std::chrono::high_resolution_clock::now();
    
    auto alloc_pool = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end_pool - start_pool);
    auto total_pool = std::chrono::duration_cast<std::chrono::microseconds>(end_pool - start_pool);
    
    std::cout << "   Allocation time: " << alloc_pool.count() << " μs" << std::endl;
    std::cout << "   Total time: " << total_pool.count() << " μs" << std::endl;
    
    // Performance comparison
    std::cout << "\n📈 Performance Results:" << std::endl;
    std::cout << "======================" << std::endl;
    
    float alloc_speedup = float(alloc_direct.count()) / float(alloc_pool.count());
    float total_speedup = float(total_direct.count()) / float(total_pool.count());
    
    std::cout << "   Allocation speedup: " << alloc_speedup << "x";
    if (alloc_speedup > 1.1f) {
        std::cout << " 🚀 FASTER!" << std::endl;
    } else if (alloc_speedup < 0.9f) {
        std::cout << " 📊 (needs optimization)" << std::endl;
    } else {
        std::cout << " ≈ (similar performance)" << std::endl;
    }
    
    std::cout << "   Total speedup: " << total_speedup << "x";
    if (total_speedup > 1.05f) {
        std::cout << " 🎯 IMPROVEMENT!" << std::endl;
    } else {
        std::cout << " (baseline)" << std::endl;
    }
    
    // Memory pool statistics
    if (auto pool_manager = device->get_memory_pool_manager()) {
        auto stats = pool_manager->get_memory_stats();
        std::cout << "\n🎯 Memory Pool Statistics:" << std::endl;
        std::cout << "   Total allocated: " << (stats.total_allocated / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Total used: " << (stats.total_used / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Active buffers: " << stats.active_buffers << std::endl;
        std::cout << "   Free buffers: " << stats.free_buffers << std::endl;
        std::cout << "   Fragmentation: " << (stats.fragmentation_ratio * 100.0f) << "%" << std::endl;
        std::cout << "   Pool count: " << stats.pool_count << std::endl;
    }
    
    std::cout << "\n✅ Phase 7.2 Memory Pool Test Complete!" << std::endl;
    
    if (alloc_speedup > 1.2f) {
        std::cout << "🚀 EXCELLENT: Memory pools provide " << ((alloc_speedup - 1.0f) * 100.0f) << "% allocation speedup!" << std::endl;
    } else if (alloc_speedup > 1.05f) {
        std::cout << "✅ GOOD: Memory pools provide " << ((alloc_speedup - 1.0f) * 100.0f) << "% allocation improvement" << std::endl;
    } else {
        std::cout << "📊 Memory pool infrastructure ready - optimization opportunities identified" << std::endl;
    }
}

int main() {
    std::cout << "DLVK Phase 7.2 Fast Memory Pool Performance Test" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    try {
        test_memory_pool_fast();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
