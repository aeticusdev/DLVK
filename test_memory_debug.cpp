#include "dlvk/core/vulkan_device.h"
#include "dlvk/core/memory_pool_manager.h"
#include <iostream>
#include <chrono>

using namespace dlvk;

int main() {
    std::cout << "🔧 DLVK Phase 7.2 Memory Pool Debug Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        // Step 1: Initialize Vulkan device
        std::cout << "1. Initializing Vulkan device..." << std::endl;
        auto device = std::make_shared<VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "❌ Failed to initialize Vulkan device" << std::endl;
            return 1;
        }
        std::cout << "✅ Vulkan device initialized: " << device->get_device_name() << std::endl;
        
        // Step 2: Test memory pool enable/disable
        std::cout << "\n2. Testing memory pool control..." << std::endl;
        std::cout << "   Initial state: " << (device->are_memory_pools_enabled() ? "enabled" : "disabled") << std::endl;
        
        device->enable_memory_pools(false);
        std::cout << "   After disable: " << (device->are_memory_pools_enabled() ? "enabled" : "disabled") << std::endl;
        
        device->enable_memory_pools(true);
        std::cout << "   After enable: " << (device->are_memory_pools_enabled() ? "enabled" : "disabled") << std::endl;
        
        // Step 3: Test direct buffer allocation
        std::cout << "\n3. Testing direct buffer allocation..." << std::endl;
        device->enable_memory_pools(false);
        
        VkBuffer test_buffer;
        VkDeviceMemory test_memory;
        
        auto start = std::chrono::high_resolution_clock::now();
        VkResult result = device->create_buffer_direct(
            1024 * 1024, // 1MB
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            test_buffer,
            test_memory
        );
        auto end = std::chrono::high_resolution_clock::now();
        
        if (result == VK_SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "✅ Direct allocation successful: " << duration.count() << " μs" << std::endl;
            device->destroy_buffer_direct(test_buffer, test_memory);
        } else {
            std::cout << "❌ Direct allocation failed: " << result << std::endl;
        }
        
        // Step 4: Test memory pool allocation
        std::cout << "\n4. Testing memory pool allocation..." << std::endl;
        device->enable_memory_pools(true);
        
        start = std::chrono::high_resolution_clock::now();
        result = device->create_buffer(
            1024 * 1024, // 1MB
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            test_buffer,
            test_memory
        );
        end = std::chrono::high_resolution_clock::now();
        
        if (result == VK_SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "✅ Pool allocation successful: " << duration.count() << " μs" << std::endl;
            device->destroy_buffer(test_buffer, test_memory);
        } else {
            std::cout << "❌ Pool allocation failed: " << result << std::endl;
        }
        
        // Step 5: Memory pool statistics
        std::cout << "\n5. Memory pool statistics..." << std::endl;
        if (auto pool_manager = device->get_memory_pool_manager()) {
            auto stats = pool_manager->get_memory_stats();
            std::cout << "   Total allocated: " << (stats.total_allocated / 1024) << " KB" << std::endl;
            std::cout << "   Total used: " << (stats.total_used / 1024) << " KB" << std::endl;
            std::cout << "   Active buffers: " << stats.active_buffers << std::endl;
            std::cout << "   Free buffers: " << stats.free_buffers << std::endl;
            std::cout << "   Pool count: " << stats.pool_count << std::endl;
        }
        
        std::cout << "\n✅ Memory pool debug test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
