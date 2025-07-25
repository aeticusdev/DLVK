#include "dlvk/core/vulkan_device.h"
#include "dlvk/core/memory_pool_manager.h"
#include <iostream>
#include <set>
#include <stdexcept>

namespace dlvk {

VulkanDevice::VulkanDevice() = default;

VulkanDevice::~VulkanDevice() {
    cleanup();
}

bool VulkanDevice::initialize() {
    if (!create_instance()) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }
    
    if (!pick_physical_device()) {
        std::cerr << "Failed to find suitable physical device" << std::endl;
        return false;
    }
    
    if (!create_logical_device()) {
        std::cerr << "Failed to create logical device" << std::endl;
        return false;
    }
    
    if (!create_command_pool()) {
        std::cerr << "Failed to create command pool" << std::endl;
        return false;
    }
    
    if (!initialize_memory_pools()) {
        std::cerr << "Failed to initialize memory pools" << std::endl;
        return false;
    }
    
    return true;
}

void VulkanDevice::cleanup() {
    // Clean up memory pool manager first
    m_memory_pool_manager.reset();
    
    if (m_command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_command_pool, nullptr);
        m_command_pool = VK_NULL_HANDLE;
    }
    
    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }
    
    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

bool VulkanDevice::create_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "DLVK ML Framework";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "DLVK";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    auto extensions = get_required_extensions();
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.enabledLayerCount = 0;

    VkResult result = vkCreateInstance(&create_info, nullptr, &m_instance);
    return result == VK_SUCCESS;
}

bool VulkanDevice::pick_physical_device() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(m_instance, &device_count, nullptr);

    if (device_count == 0) {
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(m_instance, &device_count, devices.data());

    for (const auto& device : devices) {
        if (is_device_suitable(device)) {
            m_physical_device = device;
            m_queue_families = find_queue_families(device);
            return true;
        }
    }

    return false;
}

bool VulkanDevice::create_logical_device() {
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = {
        m_queue_families.compute_family.value(),
        m_queue_families.transfer_family.value()
    };

    float queue_priority = 1.0f;
    for (uint32_t queue_family : unique_queue_families) {
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures device_features{};

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = 0;
    create_info.enabledLayerCount = 0;

    VkResult result = vkCreateDevice(m_physical_device, &create_info, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        return false;
    }

    vkGetDeviceQueue(m_device, m_queue_families.compute_family.value(), 0, &m_compute_queue);
    vkGetDeviceQueue(m_device, m_queue_families.transfer_family.value(), 0, &m_transfer_queue);

    return true;
}

bool VulkanDevice::create_command_pool() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = m_queue_families.compute_family.value();

    VkResult result = vkCreateCommandPool(m_device, &pool_info, nullptr, &m_command_pool);
    return result == VK_SUCCESS;
}

QueueFamilyIndices VulkanDevice::find_queue_families(VkPhysicalDevice device) const {
    QueueFamilyIndices indices;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    int i = 0;
    for (const auto& queue_family : queue_families) {
        if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices.compute_family = i;
        }
        
        if (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT) {
            indices.transfer_family = i;
        }

        if (indices.is_complete()) {
            break;
        }

        i++;
    }

    return indices;
}

bool VulkanDevice::is_device_suitable(VkPhysicalDevice device) const {
    QueueFamilyIndices indices = find_queue_families(device);
    return indices.is_complete();
}

std::vector<const char*> VulkanDevice::get_required_extensions() const {
    return {}; // No window extensions needed for compute-only
}

uint32_t VulkanDevice::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(m_physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

VkResult VulkanDevice::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags properties, VkBuffer& buffer,
                                   VkDeviceMemory& buffer_memory) const {
    // Use memory pool manager if available and enabled
    if (m_memory_pool_manager && m_memory_pool_manager->is_pool_enabled()) {
        return m_memory_pool_manager->allocate_buffer(size, usage, properties, buffer, buffer_memory);
    }
    
    // Fall back to direct allocation
    return create_buffer_direct(size, usage, properties, buffer, buffer_memory);
}

VkResult VulkanDevice::create_buffer_direct(VkDeviceSize size, VkBufferUsageFlags usage,
                                           VkMemoryPropertyFlags properties, VkBuffer& buffer,
                                           VkDeviceMemory& buffer_memory) const {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(m_device, &buffer_info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(m_device, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, properties);

    result = vkAllocateMemory(m_device, &alloc_info, nullptr, &buffer_memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(m_device, buffer, nullptr);
        return result;
    }

    vkBindBufferMemory(m_device, buffer, buffer_memory, 0);
    return VK_SUCCESS;
}

void VulkanDevice::destroy_buffer(VkBuffer buffer, VkDeviceMemory buffer_memory) const {
    // Use memory pool manager if available and enabled
    if (m_memory_pool_manager && m_memory_pool_manager->is_pool_enabled()) {
        m_memory_pool_manager->deallocate_buffer(buffer, buffer_memory);
        return;
    }
    
    // Fall back to direct deallocation
    destroy_buffer_direct(buffer, buffer_memory);
}

void VulkanDevice::destroy_buffer_direct(VkBuffer buffer, VkDeviceMemory buffer_memory) const {
    vkDestroyBuffer(m_device, buffer, nullptr);
    vkFreeMemory(m_device, buffer_memory, nullptr);
}

bool VulkanDevice::initialize_memory_pools() {
    try {
        // Create memory pool manager with raw pointer (no circular dependency)
        m_memory_pool_manager = std::make_unique<MemoryPoolManager>(this);
        
        std::cout << "✅ Memory pool manager initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during memory pool initialization: " << e.what() << std::endl;
        m_memory_pool_manager.reset();
        return false;
    }
}

void VulkanDevice::enable_memory_pools(bool enable) {
    if (m_memory_pool_manager) {
        m_memory_pool_manager->set_pool_enabled(enable);
        std::cout << "Memory pools " << (enable ? "enabled" : "disabled") << std::endl;
    }
}

bool VulkanDevice::are_memory_pools_enabled() const {
    return m_memory_pool_manager && m_memory_pool_manager->is_pool_enabled();
}

std::string VulkanDevice::get_device_name() const {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_physical_device, &properties);
    return std::string(properties.deviceName);
}

std::string VulkanDevice::get_device_type_string() const {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_physical_device, &properties);
    
    switch (properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_OTHER: return "Other";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "Discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "Virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
        default: return "Unknown";
    }
}

std::string VulkanDevice::get_vulkan_version_string() const {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_physical_device, &properties);
    
    uint32_t major = VK_VERSION_MAJOR(properties.apiVersion);
    uint32_t minor = VK_VERSION_MINOR(properties.apiVersion);
    uint32_t patch = VK_VERSION_PATCH(properties.apiVersion);
    
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

uint32_t VulkanDevice::get_max_workgroup_size() const {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_physical_device, &properties);
    
    // Return the maximum workgroup size across all dimensions
    return properties.limits.maxComputeWorkGroupSize[0];
}

VkDeviceSize VulkanDevice::get_max_memory_allocation() const {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_physical_device, &properties);
    return properties.limits.maxMemoryAllocationCount;
}

uint32_t VulkanDevice::get_memory_heap_count() const {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(m_physical_device, &memory_properties);
    return memory_properties.memoryHeapCount;
}

VkDeviceSize VulkanDevice::get_total_device_memory() const {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(m_physical_device, &memory_properties);
    
    VkDeviceSize total_memory = 0;
    for (uint32_t i = 0; i < memory_properties.memoryHeapCount; i++) {
        if (memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total_memory += memory_properties.memoryHeaps[i].size;
        }
    }
    return total_memory;
}

} // namespace dlvk
