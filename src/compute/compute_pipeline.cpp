#include "dlvk/compute/compute_pipeline.h"
#include "dlvk/core/vulkan_device.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace dlvk {

ComputePipeline::ComputePipeline(std::shared_ptr<VulkanDevice> device)
    : m_device(device) {}

ComputePipeline::~ComputePipeline() {
    cleanup();
}

bool ComputePipeline::create_from_file(const std::string& shader_path) {
    auto spirv_code = read_spirv_file(shader_path);
    if (spirv_code.empty()) {
        std::cerr << "Failed to read SPIR-V file: " << shader_path << std::endl;
        return false;
    }
    
    return create_from_spirv(spirv_code);
}

bool ComputePipeline::create_from_spirv(const std::vector<uint32_t>& spirv_code) {
    if (!create_shader_module(spirv_code)) {
        std::cerr << "Failed to create shader module" << std::endl;
        return false;
    }
    
    if (!create_pipeline_layout()) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        return false;
    }
    
    if (!create_pipeline()) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        return false;
    }
    
    return true;
}

bool ComputePipeline::create_descriptor_set_layout(const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    
    VkResult result = vkCreateDescriptorSetLayout(m_device->get_device(), &layout_info, 
                                                 nullptr, &m_descriptor_set_layout);
    return result == VK_SUCCESS;
}

bool ComputePipeline::allocate_descriptor_sets(uint32_t count) {
    if (m_descriptor_set_layout == VK_NULL_HANDLE) {
        std::cerr << "Descriptor set layout must be created first" << std::endl;
        return false;
    }
    

    std::vector<VkDescriptorSetLayoutBinding> bindings;

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = count * 4; // Assume max 4 buffers per set
    
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = count;
    
    VkResult result = vkCreateDescriptorPool(m_device->get_device(), &pool_info, 
                                           nullptr, &m_descriptor_pool);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        return false;
    }
    

    std::vector<VkDescriptorSetLayout> layouts(count, m_descriptor_set_layout);
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_descriptor_pool;
    alloc_info.descriptorSetCount = count;
    alloc_info.pSetLayouts = layouts.data();
    
    m_descriptor_sets.resize(count);
    result = vkAllocateDescriptorSets(m_device->get_device(), &alloc_info, m_descriptor_sets.data());
    
    return result == VK_SUCCESS;
}

void ComputePipeline::update_descriptor_set(uint32_t set_index, uint32_t binding, 
                                           VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
    if (set_index >= m_descriptor_sets.size()) {
        std::cerr << "Invalid descriptor set index" << std::endl;
        return;
    }
    
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer;
    buffer_info.offset = offset;
    buffer_info.range = range;
    
    VkWriteDescriptorSet descriptor_write{};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = m_descriptor_sets[set_index];
    descriptor_write.dstBinding = binding;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_write.descriptorCount = 1;
    descriptor_write.pBufferInfo = &buffer_info;
    
    vkUpdateDescriptorSets(m_device->get_device(), 1, &descriptor_write, 0, nullptr);
}

void ComputePipeline::set_push_constant_range(const PushConstantRange& range) {
    m_push_constant_range = range;
    m_has_push_constants = true;
}

void ComputePipeline::bind(VkCommandBuffer cmd_buffer) {
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    
    if (!m_descriptor_sets.empty()) {
        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
                               m_pipeline_layout, 0, 1, &m_descriptor_sets[0], 0, nullptr);
    }
}

void ComputePipeline::dispatch(VkCommandBuffer cmd_buffer, uint32_t group_count_x, 
                              uint32_t group_count_y, uint32_t group_count_z) {
    vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, group_count_z);
}

void ComputePipeline::push_constants(VkCommandBuffer cmd_buffer, const void* data, 
                                    uint32_t size, uint32_t offset) {
    if (!m_has_push_constants) {
        std::cerr << "Push constants not configured for this pipeline" << std::endl;
        return;
    }
    
    vkCmdPushConstants(cmd_buffer, m_pipeline_layout, m_push_constant_range.stage_flags, 
                      m_push_constant_range.offset + offset, size, data);
}

VkDescriptorSet ComputePipeline::get_descriptor_set(uint32_t index) const {
    if (index >= m_descriptor_sets.size()) {
        return VK_NULL_HANDLE;
    }
    return m_descriptor_sets[index];
}

void ComputePipeline::cleanup() {
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device->get_device(), m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    
    if (m_pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device->get_device(), m_pipeline_layout, nullptr);
        m_pipeline_layout = VK_NULL_HANDLE;
    }
    
    if (m_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device->get_device(), m_descriptor_pool, nullptr);
        m_descriptor_pool = VK_NULL_HANDLE;
    }
    
    if (m_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device->get_device(), m_descriptor_set_layout, nullptr);
        m_descriptor_set_layout = VK_NULL_HANDLE;
    }
    
    if (m_shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device->get_device(), m_shader_module, nullptr);
        m_shader_module = VK_NULL_HANDLE;
    }
}

bool ComputePipeline::create_shader_module(const std::vector<uint32_t>& spirv_code) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    create_info.pCode = spirv_code.data();
    
    VkResult result = vkCreateShaderModule(m_device->get_device(), &create_info, 
                                          nullptr, &m_shader_module);
    return result == VK_SUCCESS;
}

bool ComputePipeline::create_pipeline_layout() {
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    
    if (m_descriptor_set_layout != VK_NULL_HANDLE) {
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &m_descriptor_set_layout;
    }
    
    if (m_has_push_constants) {
        VkPushConstantRange push_constant{};
        push_constant.stageFlags = m_push_constant_range.stage_flags;
        push_constant.offset = m_push_constant_range.offset;
        push_constant.size = m_push_constant_range.size;
        
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant;
    }
    
    VkResult result = vkCreatePipelineLayout(m_device->get_device(), &pipeline_layout_info, 
                                           nullptr, &m_pipeline_layout);
    return result == VK_SUCCESS;
}

bool ComputePipeline::create_pipeline() {
    VkPipelineShaderStageCreateInfo shader_stage_info{};
    shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_info.module = m_shader_module;
    shader_stage_info.pName = "main";
    
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = m_pipeline_layout;
    pipeline_info.stage = shader_stage_info;
    
    VkResult result = vkCreateComputePipelines(m_device->get_device(), VK_NULL_HANDLE, 
                                              1, &pipeline_info, nullptr, &m_pipeline);
    return result == VK_SUCCESS;
}

std::vector<uint32_t> ComputePipeline::read_spirv_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }
    
    size_t file_size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));
    
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    file.close();
    
    return buffer;
}

} // namespace dlvk
