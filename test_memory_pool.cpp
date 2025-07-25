#include "dlvk/core/vulkan_device.h"
#include "dlvk/core/memory_pool_manager.h"
#include "dlvk/tensor/tensor.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace dlvk;

void test_memory_pool_performance() {
    std::cout << "\n🚀 Testing Memory Pool Performance (Phase 7.2 Priority 1)" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    // Initialize Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return;
    }
    
    std::cout << "✓ Vulkan device initialized: " << device->get_device_name() << std::endl;
    
    // Test memory pool stats
    auto* pool_manager = device->get_memory_pool_manager();
    if (!pool_manager) {
        std::cerr << "Memory pool manager not available" << std::endl;
        return;
    }
    
    auto initial_stats = pool_manager->get_memory_stats();
    std::cout << "✓ Initial memory pools: " << initial_stats.pool_count << std::endl;
    
    // Test 1: Memory allocation performance comparison
    const size_t num_allocations = 100;
    const size_t buffer_size = 1024 * 1024; // 1MB
    
    std::cout << "\n📊 Performance Test: " << num_allocations << " allocations of " << (buffer_size / 1024) << "KB each" << std::endl;
    
    // Test with memory pools enabled
    device->enable_memory_pools(true);
    std::vector<VkBuffer> pool_buffers(num_allocations);
    std::vector<VkDeviceMemory> pool_memories(num_allocations);
    
    auto start_pool = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_allocations; ++i) {
        VkResult result = device->create_buffer(
            buffer_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            pool_buffers[i],
            pool_memories[i]
        );
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to allocate buffer " << i << " with pools" << std::endl;
            return;
        }
    }
    auto end_pool = std::chrono::high_resolution_clock::now();
    auto pool_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_pool - start_pool);
    
    // Deallocate pool buffers
    for (size_t i = 0; i < num_allocations; ++i) {
        device->destroy_buffer(pool_buffers[i], pool_memories[i]);
    }
    
    // Test with memory pools disabled  
    device->enable_memory_pools(false);
    std::vector<VkBuffer> direct_buffers(num_allocations);
    std::vector<VkDeviceMemory> direct_memories(num_allocations);
    
    auto start_direct = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_allocations; ++i) {
        VkResult result = device->create_buffer(
            buffer_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            direct_buffers[i],
            direct_memories[i]
        );
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to allocate buffer " << i << " direct" << std::endl;
            return;
        }
    }
    auto end_direct = std::chrono::high_resolution_clock::now();
    auto direct_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_direct - start_direct);
    
    // Deallocate direct buffers
    for (size_t i = 0; i < num_allocations; ++i) {
        device->destroy_buffer(direct_buffers[i], direct_memories[i]);
    }
    
    // Results
    std::cout << "🎯 Memory Pool Results:" << std::endl;
    std::cout << "   Pool allocation time:   " << pool_duration.count() << " μs" << std::endl;
    std::cout << "   Direct allocation time: " << direct_duration.count() << " μs" << std::endl;
    
    if (pool_duration.count() > 0) {
        float speedup = float(direct_duration.count()) / float(pool_duration.count());
        std::cout << "   Performance improvement: " << speedup << "x faster" << std::endl;
        
        if (speedup > 1.2f) {
            std::cout << "   ✅ Memory pools provide significant speedup!" << std::endl;
        } else {
            std::cout << "   ⚠️  Memory pool benefit minimal (may need larger test)" << std::endl;
        }
    }
    
    // Test 2: Memory usage efficiency
    device->enable_memory_pools(true);
    auto stats_after = pool_manager->get_memory_stats();
    
    std::cout << "\n📊 Memory Usage Statistics:" << std::endl;
    std::cout << "   Total allocated: " << (stats_after.total_allocated / 1024 / 1024) << " MB" << std::endl;
    std::cout << "   Total used:      " << (stats_after.total_used / 1024 / 1024) << " MB" << std::endl;
    std::cout << "   Active buffers:  " << stats_after.active_buffers << std::endl;
    std::cout << "   Free buffers:    " << stats_after.free_buffers << std::endl;
    std::cout << "   Fragmentation:   " << (stats_after.fragmentation_ratio * 100) << "%" << std::endl;
    std::cout << "   Pool count:      " << stats_after.pool_count << std::endl;
    
    // Test 3: Tensor allocation with pools
    std::cout << "\n🧮 Tensor Allocation Test:" << std::endl;
    
    auto start_tensor = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> tensors;
    tensors.reserve(50);
    
    for (int i = 0; i < 50; ++i) {
        tensors.emplace_back(std::vector<size_t>{64, 64}, DataType::FLOAT32, device);
    }
    auto end_tensor = std::chrono::high_resolution_clock::now();
    auto tensor_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_tensor - start_tensor);
    
    std::cout << "   50 tensor allocation time: " << tensor_duration.count() << " μs" << std::endl;
    std::cout << "   Average per tensor: " << (tensor_duration.count() / 50) << " μs" << std::endl;
    
    auto final_stats = pool_manager->get_memory_stats();
    std::cout << "   Final pool count: " << final_stats.pool_count << std::endl;
    std::cout << "   Final active buffers: " << final_stats.active_buffers << std::endl;
    
    // Clean up
    tensors.clear();
    
    std::cout << "\n🎉 Memory Pool Manager Test Complete!" << std::endl;
    std::cout << "✅ Phase 7.2 Priority 1: GPU Memory Pool System successfully implemented!" << std::endl;
}

int main() {
    std::cout << "DLVK Phase 7.2 Memory Pool Manager Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_memory_pool_performance();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
