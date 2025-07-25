#include "dlvk/core/vulkan_device.h"
#include "dlvk/core/memory_pool_manager.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace dlvk;

void test_memory_pool_with_compute() {
    std::cout << "\n🚀 DLVK Phase 7.2 Memory Pool + Compute Performance Test" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    // Initialize Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return;
    }
    
    std::cout << "✓ Vulkan device initialized: " << device->get_device_name() << std::endl;
    
    // Test 1: Memory allocation without pools (baseline)
    std::cout << "\n📊 Test 1: Direct Allocation + Compute (Baseline)" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    device->enable_memory_pools(false);
    std::cout << "   Memory pools: DISABLED" << std::endl;
    
    auto start_baseline = std::chrono::high_resolution_clock::now();
    
    // Create tensors for matrix operations
    std::vector<Tensor> input_tensors;
    std::vector<Tensor> output_tensors;
    
    for (int i = 0; i < 20; ++i) {
        input_tensors.emplace_back(std::vector<size_t>{512, 512}, DataType::FLOAT32, device);
        output_tensors.emplace_back(std::vector<size_t>{512, 512}, DataType::FLOAT32, device);
        
        // Fill with test data
        std::vector<float> data(512 * 512, 1.0f + i * 0.1f);
        input_tensors.back().upload_data(data.data());
    }
    
    auto alloc_end_baseline = std::chrono::high_resolution_clock::now();
    auto alloc_baseline = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end_baseline - start_baseline);
    
    // Perform compute operations
    TensorOps tensor_ops(device);
    if (!tensor_ops.initialize()) {
        std::cerr << "Failed to initialize TensorOps" << std::endl;
        return;
    }
    
    auto compute_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; ++i) {
        // Matrix multiply operations (will use GPU compute shaders!)
        tensor_ops.matrix_multiply(input_tensors[i], input_tensors[i + 1], output_tensors[i]);
        
        // Element-wise operations (will use GPU compute shaders!)
        tensor_ops.add(output_tensors[i], input_tensors[i + 2], output_tensors[i]);
    }
    
    // Wait for all operations to complete
    vkDeviceWaitIdle(device->get_device());
    
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_baseline = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start);
    
    auto total_baseline = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - start_baseline);
    
    std::cout << "   Allocation time: " << alloc_baseline.count() << " μs" << std::endl;
    std::cout << "   Compute time: " << compute_baseline.count() << " μs" << std::endl;
    std::cout << "   Total time: " << total_baseline.count() << " μs" << std::endl;
    
    // Clear tensors
    input_tensors.clear();
    output_tensors.clear();
    
    // Test 2: Memory allocation with pools (optimized)
    std::cout << "\n🎯 Test 2: Memory Pool Allocation + Compute (Optimized)" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    device->enable_memory_pools(true);
    std::cout << "   Memory pools: ENABLED" << std::endl;
    
    auto start_optimized = std::chrono::high_resolution_clock::now();
    
    // Create tensors with memory pools
    std::vector<Tensor> pool_input_tensors;
    std::vector<Tensor> pool_output_tensors;
    
    for (int i = 0; i < 20; ++i) {
        pool_input_tensors.emplace_back(std::vector<size_t>{512, 512}, DataType::FLOAT32, device);
        pool_output_tensors.emplace_back(std::vector<size_t>{512, 512}, DataType::FLOAT32, device);
        
        // Fill with test data
        std::vector<float> data(512 * 512, 1.0f + i * 0.1f);
        pool_input_tensors.back().upload_data(data.data());
    }
    
    auto alloc_end_optimized = std::chrono::high_resolution_clock::now();
    auto alloc_optimized = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end_optimized - start_optimized);
    
    // Perform same compute operations
    auto compute_start_opt = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; ++i) {
        // Matrix multiply operations (reusing same TensorOps instance)
        tensor_ops.matrix_multiply(pool_input_tensors[i], pool_input_tensors[i + 1], pool_output_tensors[i]);
        
        // Element-wise operations (reusing same TensorOps instance)
        tensor_ops.add(pool_output_tensors[i], pool_input_tensors[i + 2], pool_output_tensors[i]);
    }
    
    // Wait for all operations to complete
    vkDeviceWaitIdle(device->get_device());
    
    auto compute_end_opt = std::chrono::high_resolution_clock::now();
    auto compute_optimized = std::chrono::duration_cast<std::chrono::microseconds>(compute_end_opt - compute_start_opt);
    
    auto total_optimized = std::chrono::duration_cast<std::chrono::microseconds>(compute_end_opt - start_optimized);
    
    std::cout << "   Allocation time: " << alloc_optimized.count() << " μs" << std::endl;
    std::cout << "   Compute time: " << compute_optimized.count() << " μs" << std::endl;
    std::cout << "   Total time: " << total_optimized.count() << " μs" << std::endl;
    
    // Performance comparison
    std::cout << "\n📈 Performance Comparison:" << std::endl;
    std::cout << "=========================" << std::endl;
    
    float alloc_speedup = float(alloc_baseline.count()) / float(alloc_optimized.count());
    float total_speedup = float(total_baseline.count()) / float(total_optimized.count());
    
    std::cout << "   Allocation speedup: " << alloc_speedup << "x" << std::endl;
    std::cout << "   Total speedup: " << total_speedup << "x" << std::endl;
    std::cout << "   Memory pool efficiency: " << ((alloc_speedup - 1.0f) * 100.0f) << "% improvement" << std::endl;
    
    // Memory pool statistics
    if (auto pool_manager = device->get_memory_pool_manager()) {
        auto stats = pool_manager->get_memory_stats();
        std::cout << "\n🎯 Memory Pool Statistics:" << std::endl;
        std::cout << "   Total allocated: " << (stats.total_allocated / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Total used: " << (stats.total_used / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Active buffers: " << stats.active_buffers << std::endl;
        std::cout << "   Free buffers: " << stats.free_buffers << std::endl;
        std::cout << "   Fragmentation ratio: " << (stats.fragmentation_ratio * 100.0f) << "%" << std::endl;
        std::cout << "   Pool count: " << stats.pool_count << std::endl;
    }
    
    std::cout << "\n✅ Phase 7.2 Memory Pool Performance Test Complete!" << std::endl;
    if (total_speedup > 1.1f) {
        std::cout << "🚀 Memory pools provide significant performance improvement!" << std::endl;
    } else {
        std::cout << "📊 Memory pools ready for optimization - further tuning needed" << std::endl;
    }
}

void test_rapid_allocation_deallocation() {
    std::cout << "\n🔄 Rapid Allocation/Deallocation Test (Memory Pool Stress Test)" << std::endl;
    std::cout << "==============================================================" << std::endl;
    
    auto device = std::make_shared<VulkanDevice>();
    device->initialize();
    
    // Test without pools
    device->enable_memory_pools(false);
    auto start_direct = std::chrono::high_resolution_clock::now();
    
    for (int cycle = 0; cycle < 5; ++cycle) {
        std::vector<Tensor> temp_tensors;
        for (int i = 0; i < 50; ++i) {
            temp_tensors.emplace_back(std::vector<size_t>{128, 128}, DataType::FLOAT32, device);
        }
        // Tensors automatically destroyed when going out of scope
    }
    
    auto end_direct = std::chrono::high_resolution_clock::now();
    auto duration_direct = std::chrono::duration_cast<std::chrono::microseconds>(end_direct - start_direct);
    
    // Test with pools
    device->enable_memory_pools(true);
    auto start_pool = std::chrono::high_resolution_clock::now();
    
    for (int cycle = 0; cycle < 5; ++cycle) {
        std::vector<Tensor> temp_tensors;
        for (int i = 0; i < 50; ++i) {
            temp_tensors.emplace_back(std::vector<size_t>{128, 128}, DataType::FLOAT32, device);
        }
        // Tensors automatically destroyed when going out of scope
    }
    
    auto end_pool = std::chrono::high_resolution_clock::now();
    auto duration_pool = std::chrono::duration_cast<std::chrono::microseconds>(end_pool - start_pool);
    
    std::cout << "   Direct allocation: " << duration_direct.count() << " μs" << std::endl;
    std::cout << "   Pool allocation: " << duration_pool.count() << " μs" << std::endl;
    std::cout << "   Speedup: " << (float(duration_direct.count()) / float(duration_pool.count())) << "x" << std::endl;
}

int main() {
    std::cout << "DLVK Phase 7.2 Memory Pool + GPU Compute Performance Test" << std::endl;
    std::cout << "==========================================================" << std::endl;
    
    try {
        test_memory_pool_with_compute();
        test_rapid_allocation_deallocation();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
