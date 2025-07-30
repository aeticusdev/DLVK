#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>

namespace dlvk {

class VulkanDevice;

struct PushConstantRange {
    uint32_t offset;
    uint32_t size;
    VkShaderStageFlags stage_flags;
};

class ComputePipeline {
public:
    ComputePipeline(std::shared_ptr<VulkanDevice> device);
    ~ComputePipeline();


    bool create_from_file(const std::string& shader_path);
    bool create_from_spirv(const std::vector<uint32_t>& spirv_code);
    

    bool create_descriptor_set_layout(const std::vector<VkDescriptorSetLayoutBinding>& bindings);
    bool allocate_descriptor_sets(uint32_t count = 1);
    void update_descriptor_set(uint32_t set_index, uint32_t binding, VkBuffer buffer, 
                              VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
    

    void set_push_constant_range(const PushConstantRange& range);
    

    void bind(VkCommandBuffer cmd_buffer);
    void dispatch(VkCommandBuffer cmd_buffer, uint32_t group_count_x, 
                 uint32_t group_count_y = 1, uint32_t group_count_z = 1);
    void push_constants(VkCommandBuffer cmd_buffer, const void* data, 
                       uint32_t size, uint32_t offset = 0);
    

    void cleanup();
    

    VkPipeline get_pipeline() const { return m_pipeline; }
    VkPipelineLayout get_layout() const { return m_pipeline_layout; }
    VkDescriptorSetLayout get_descriptor_layout() const { return m_descriptor_set_layout; }
    VkDescriptorSet get_descriptor_set(uint32_t index = 0) const;

private:
    std::shared_ptr<VulkanDevice> m_device;
    
    VkShaderModule m_shader_module = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_descriptor_sets;
    
    PushConstantRange m_push_constant_range = {};
    bool m_has_push_constants = false;
    

    bool create_shader_module(const std::vector<uint32_t>& spirv_code);
    bool create_pipeline_layout();
    bool create_pipeline();
    bool create_descriptor_pool(const std::vector<VkDescriptorSetLayoutBinding>& bindings, uint32_t set_count);
    
    std::vector<uint32_t> read_spirv_file(const std::string& filename);
};

} // namespace dlvk
