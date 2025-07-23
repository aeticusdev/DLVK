#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <string>

namespace dlvk {

struct QueueFamilyIndices {
    std::optional<uint32_t> compute_family;
    std::optional<uint32_t> transfer_family;
    
    bool is_complete() const {
        return compute_family.has_value() && transfer_family.has_value();
    }
};

class VulkanDevice {
public:
    VulkanDevice();
    ~VulkanDevice();

    bool initialize();
    void cleanup();

    VkDevice get_device() const { return m_device; }
    VkPhysicalDevice get_physical_device() const { return m_physical_device; }
    VkInstance get_instance() const { return m_instance; }
    VkQueue get_compute_queue() const { return m_compute_queue; }
    VkQueue get_transfer_queue() const { return m_transfer_queue; }
    
    const QueueFamilyIndices& get_queue_families() const { return m_queue_families; }
    
    VkCommandPool get_command_pool() const { return m_command_pool; }
    
    // Memory operations
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;
    
    // Buffer operations
    VkResult create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                          VkMemoryPropertyFlags properties, VkBuffer& buffer, 
                          VkDeviceMemory& buffer_memory) const;
    
    void destroy_buffer(VkBuffer buffer, VkDeviceMemory buffer_memory) const;
    
    // Device information
    std::string get_device_name() const;
    std::string get_device_type_string() const;
    std::string get_vulkan_version_string() const;
    uint32_t get_max_workgroup_size() const;
    VkDeviceSize get_max_memory_allocation() const;
    uint32_t get_memory_heap_count() const;
    VkDeviceSize get_total_device_memory() const;

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_compute_queue = VK_NULL_HANDLE;
    VkQueue m_transfer_queue = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    
    QueueFamilyIndices m_queue_families;
    
    bool create_instance();
    bool pick_physical_device();
    bool create_logical_device();
    bool create_command_pool();
    
    QueueFamilyIndices find_queue_families(VkPhysicalDevice device) const;
    bool is_device_suitable(VkPhysicalDevice device) const;
    std::vector<const char*> get_required_extensions() const;
};

} // namespace dlvk
