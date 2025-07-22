#include <iostream>
#include "dlvk/core/vulkan_device.h"

int main() {
    std::cout << "Testing Vulkan device initialization...\n";
    
    try {
        dlvk::VulkanDevice device;
        
        if (device.initialize()) {
            std::cout << "✓ Vulkan device initialized successfully\n";
            std::cout << "✓ Instance: " << (device.get_instance() ? "Valid" : "Invalid") << std::endl;
            std::cout << "✓ Device: " << (device.get_device() ? "Valid" : "Invalid") << std::endl;
            std::cout << "✓ Physical device: " << (device.get_physical_device() ? "Valid" : "Invalid") << std::endl;
            std::cout << "✓ Compute queue: " << (device.get_compute_queue() ? "Valid" : "Invalid") << std::endl;
            
            // Test buffer creation
            VkBuffer buffer;
            VkDeviceMemory memory;
            
            VkResult result = device.create_buffer(
                1024,  // 1KB buffer
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                buffer,
                memory
            );
            
            if (result == VK_SUCCESS) {
                std::cout << "✓ Buffer creation test passed\n";
                device.destroy_buffer(buffer, memory);
                std::cout << "✓ Buffer destruction test passed\n";
            } else {
                std::cout << "✗ Buffer creation test failed\n";
                return -1;
            }
            
            std::cout << "All Vulkan device tests passed!\n";
            
        } else {
            std::cout << "✗ Failed to initialize Vulkan device\n";
            std::cout << "This might be due to:\n";
            std::cout << "- No Vulkan-capable GPU\n";
            std::cout << "- Missing Vulkan drivers\n";
            std::cout << "- Incompatible Vulkan version\n";
            return -1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
