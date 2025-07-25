#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace dlvk;

void test_basic_allocation_performance() {
    std::cout << "\n🚀 Testing Basic Memory Allocation Performance (Phase 7.2)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    
    // Initialize Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return;
    }
    
    std::cout << "✓ Vulkan device initialized: " << device->get_device_name() << std::endl;
    
    // Test 1: Direct buffer allocation performance
    const size_t num_allocations = 50;
    const size_t buffer_size = 1024 * 1024; // 1MB
    
    std::cout << "\n📊 Direct Buffer Allocation Test: " << num_allocations << " allocations of " << (buffer_size / 1024) << "KB each" << std::endl;
    
    std::vector<VkBuffer> buffers(num_allocations);
    std::vector<VkDeviceMemory> memories(num_allocations);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_allocations; ++i) {
        VkResult result = device->create_buffer_direct(
            buffer_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            buffers[i],
            memories[i]
        );
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to allocate buffer " << i << std::endl;
            return;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "   Total allocation time: " << duration.count() << " μs" << std::endl;
    std::cout << "   Average per buffer: " << (duration.count() / num_allocations) << " μs" << std::endl;
    std::cout << "   Allocation rate: " << (num_allocations * 1000000.0 / duration.count()) << " buffers/second" << std::endl;
    
    // Deallocate buffers
    auto dealloc_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_allocations; ++i) {
        device->destroy_buffer_direct(buffers[i], memories[i]);
    }
    auto dealloc_end = std::chrono::high_resolution_clock::now();
    auto dealloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(dealloc_end - dealloc_start);
    
    std::cout << "   Total deallocation time: " << dealloc_duration.count() << " μs" << std::endl;
    std::cout << "   Average per buffer: " << (dealloc_duration.count() / num_allocations) << " μs" << std::endl;
    
    // Test 2: Tensor allocation performance
    std::cout << "\n🧮 Tensor Allocation Test:" << std::endl;
    
    auto tensor_start = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> tensors;
    tensors.reserve(30);
    
    for (int i = 0; i < 30; ++i) {
        tensors.emplace_back(std::vector<size_t>{256, 256}, DataType::FLOAT32, device);
    }
    auto tensor_end = std::chrono::high_resolution_clock::now();
    auto tensor_duration = std::chrono::duration_cast<std::chrono::microseconds>(tensor_end - tensor_start);
    
    std::cout << "   30 tensor allocation time: " << tensor_duration.count() << " μs" << std::endl;
    std::cout << "   Average per tensor: " << (tensor_duration.count() / 30) << " μs" << std::endl;
    std::cout << "   Tensor size: 256x256 floats = " << (256 * 256 * 4 / 1024) << "KB each" << std::endl;
    
    // Test 3: Memory allocation patterns analysis
    std::cout << "\n📈 Memory Pattern Analysis:" << std::endl;
    
    // Small allocations
    auto small_start = std::chrono::high_resolution_clock::now();
    std::vector<VkBuffer> small_buffers(100);
    std::vector<VkDeviceMemory> small_memories(100);
    
    for (size_t i = 0; i < 100; ++i) {
        device->create_buffer_direct(
            4096, // 4KB
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            small_buffers[i],
            small_memories[i]
        );
    }
    auto small_end = std::chrono::high_resolution_clock::now();
    auto small_duration = std::chrono::duration_cast<std::chrono::microseconds>(small_end - small_start);
    
    // Large allocations
    auto large_start = std::chrono::high_resolution_clock::now();
    std::vector<VkBuffer> large_buffers(10);
    std::vector<VkDeviceMemory> large_memories(10);
    
    for (size_t i = 0; i < 10; ++i) {
        device->create_buffer_direct(
            16 * 1024 * 1024, // 16MB
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            large_buffers[i],
            large_memories[i]
        );
    }
    auto large_end = std::chrono::high_resolution_clock::now();
    auto large_duration = std::chrono::duration_cast<std::chrono::microseconds>(large_end - large_start);
    
    std::cout << "   Small allocations (4KB x 100): " << small_duration.count() << " μs" << std::endl;
    std::cout << "   Large allocations (16MB x 10): " << large_duration.count() << " μs" << std::endl;
    std::cout << "   Small vs Large efficiency: " << (float(small_duration.count()) / 100.0f) << " vs " << (float(large_duration.count()) / 10.0f) << " μs per allocation" << std::endl;
    
    // Cleanup
    for (size_t i = 0; i < 100; ++i) {
        device->destroy_buffer_direct(small_buffers[i], small_memories[i]);
    }
    for (size_t i = 0; i < 10; ++i) {
        device->destroy_buffer_direct(large_buffers[i], large_memories[i]);
    }
    tensors.clear();
    
    std::cout << "\n📊 Performance Analysis Summary:" << std::endl;
    std::cout << "   Current allocation method: Direct Vulkan allocation per buffer" << std::endl;
    std::cout << "   Optimization opportunity: Memory pooling can reduce allocation overhead" << std::endl;
    std::cout << "   Expected improvement: 2-5x faster for frequent small allocations" << std::endl;
    
    std::cout << "\n✅ Phase 7.2 Baseline Performance Test Complete!" << std::endl;
    std::cout << "🎯 Next step: Implement memory pool optimization for improved performance" << std::endl;
}

int main() {
    std::cout << "DLVK Phase 7.2 Memory Performance Baseline Test" << std::endl;
    std::cout << "================================================" << std::endl;
    
    try {
        test_basic_allocation_performance();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
