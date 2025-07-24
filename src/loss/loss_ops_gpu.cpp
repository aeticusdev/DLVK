#include "dlvk/loss/loss_ops_gpu.h"
#include "dlvk/compute/compute_pipeline.h"
#include "dlvk/core/vulkan_device.h"
#include <iostream>

namespace dlvk {

LossOpsGPU::LossOpsGPU(std::shared_ptr<VulkanDevice> device) : m_device(device) {}

LossOpsGPU::~LossOpsGPU() {
    if (m_fence != VK_NULL_HANDLE) {
        vkDestroyFence(m_device->get_device(), m_fence, nullptr);
    }
    
    if (m_command_buffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device->get_device(), m_device->get_command_pool(), 1, &m_command_buffer);
    }
}

bool LossOpsGPU::initialize() {
    if (!allocate_command_buffer()) {
        std::cerr << "Failed to allocate command buffer for loss operations" << std::endl;
        return false;
    }
    
    // Create fence for synchronization
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    
    VkResult result = vkCreateFence(m_device->get_device(), &fence_info, nullptr, &m_fence);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create fence for loss operations" << std::endl;
        return false;
    }
    
    if (!create_pipelines()) {
        std::cerr << "Failed to create loss compute pipelines" << std::endl;
        return false;
    }
    
    std::cout << "✓ LossOpsGPU initialized successfully" << std::endl;
    return true;
}

bool LossOpsGPU::allocate_command_buffer() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = m_device->get_command_pool();
    alloc_info.commandBufferCount = 1;
    
    VkResult result = vkAllocateCommandBuffers(m_device->get_device(), &alloc_info, &m_command_buffer);
    return result == VK_SUCCESS;
}

bool LossOpsGPU::create_pipelines() {
    // Setup descriptor set layout for loss functions (3 buffers: input1, input2, output)
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }
    
    // Setup push constant range
    PushConstantRange push_range;
    push_range.offset = 0;
    push_range.size = sizeof(uint32_t) * 4;  // size, batch_size, and additional params
    push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    int success_count = 0;
    
    // Create MSE forward pipeline
    m_mse_forward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_mse_forward_pipeline->create_descriptor_set_layout(bindings)) {
        m_mse_forward_pipeline->set_push_constant_range(push_range);
        if (m_mse_forward_pipeline->create_from_file("build/shaders/mse_forward.comp.spv")) {
            if (m_mse_forward_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ MSE forward pipeline created successfully" << std::endl;
                success_count++;
            } else {
                std::cout << "✗ MSE forward pipeline descriptor allocation failed" << std::endl;
                m_mse_forward_pipeline.reset();
            }
        } else {
            std::cout << "✗ MSE forward pipeline shader loading failed" << std::endl;
            m_mse_forward_pipeline.reset();
        }
    } else {
        std::cout << "✗ MSE forward pipeline descriptor layout creation failed" << std::endl;
        m_mse_forward_pipeline.reset();
    }
    
    // Create MSE backward pipeline
    m_mse_backward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_mse_backward_pipeline->create_descriptor_set_layout(bindings)) {
        m_mse_backward_pipeline->set_push_constant_range(push_range);
        if (m_mse_backward_pipeline->create_from_file("build/shaders/mse_backward.comp.spv")) {
            if (m_mse_backward_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ MSE backward pipeline created successfully" << std::endl;
                success_count++;
            } else {
                std::cout << "✗ MSE backward pipeline descriptor allocation failed" << std::endl;
                m_mse_backward_pipeline.reset();
            }
        } else {
            std::cout << "✗ MSE backward pipeline shader loading failed" << std::endl;
            m_mse_backward_pipeline.reset();
        }
    }
    
    std::cout << "Created " << success_count << " loss function pipelines successfully" << std::endl;
    return success_count > 0;
}

VkCommandBuffer LossOpsGPU::begin_single_time_commands() {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(m_command_buffer, &begin_info);
    return m_command_buffer;
}

void LossOpsGPU::end_single_time_commands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    
    vkQueueSubmit(m_device->get_compute_queue(), 1, &submit_info, m_fence);
    vkWaitForFences(m_device->get_device(), 1, &m_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(m_device->get_device(), 1, &m_fence);
    vkResetCommandBuffer(cmd, 0);
}

bool LossOpsGPU::mse_forward(const std::shared_ptr<Tensor>& predictions, 
                             const std::shared_ptr<Tensor>& targets,
                             std::shared_ptr<Tensor>& result) {
    if (!m_mse_forward_pipeline) {
        std::cerr << "MSE forward pipeline not available" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_mse_forward_pipeline->update_descriptor_set(0, 0, predictions->buffer());
    m_mse_forward_pipeline->update_descriptor_set(0, 1, targets->buffer());
    m_mse_forward_pipeline->update_descriptor_set(0, 2, result->buffer());
    
    // Bind pipeline
    m_mse_forward_pipeline->bind(cmd);
    
    // Push constants
    struct MSEPushConstants {
        uint32_t size;
        uint32_t batch_size;
    } push_constants;
    
    push_constants.size = static_cast<uint32_t>(predictions->size());
    push_constants.batch_size = static_cast<uint32_t>(predictions->shape()[0]);
    
    m_mse_forward_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    
    // Dispatch compute shader (256 threads per workgroup)
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (push_constants.size + workgroup_size - 1) / workgroup_size;
    m_mse_forward_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool LossOpsGPU::mse_backward(const std::shared_ptr<Tensor>& predictions,
                              const std::shared_ptr<Tensor>& targets,
                              std::shared_ptr<Tensor>& gradient) {
    if (!m_mse_backward_pipeline) {
        std::cerr << "MSE backward pipeline not available" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_mse_backward_pipeline->update_descriptor_set(0, 0, predictions->buffer());
    m_mse_backward_pipeline->update_descriptor_set(0, 1, targets->buffer());
    m_mse_backward_pipeline->update_descriptor_set(0, 2, gradient->buffer());
    
    // Bind pipeline
    m_mse_backward_pipeline->bind(cmd);
    
    // Push constants
    struct MSEBackwardPushConstants {
        uint32_t size;
        float scale_factor;
    } push_constants;
    
    push_constants.size = static_cast<uint32_t>(predictions->size());
    push_constants.scale_factor = 2.0f / static_cast<float>(predictions->size());
    
    m_mse_backward_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    
    // Dispatch compute shader
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (push_constants.size + workgroup_size - 1) / workgroup_size;
    m_mse_backward_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

// Placeholder implementations for other loss functions
bool LossOpsGPU::cross_entropy_forward(const std::shared_ptr<Tensor>& predictions,
                                        const std::shared_ptr<Tensor>& targets,
                                        std::shared_ptr<Tensor>& result) {
    // TODO: Implement when cross entropy pipeline is ready
    std::cerr << "Cross entropy forward GPU implementation not yet ready" << std::endl;
    return false;
}

bool LossOpsGPU::cross_entropy_backward(const std::shared_ptr<Tensor>& predictions,
                                         const std::shared_ptr<Tensor>& targets,
                                         std::shared_ptr<Tensor>& gradient) {
    // TODO: Implement when cross entropy pipeline is ready
    std::cerr << "Cross entropy backward GPU implementation not yet ready" << std::endl;
    return false;
}

bool LossOpsGPU::binary_cross_entropy_forward(const std::shared_ptr<Tensor>& predictions,
                                               const std::shared_ptr<Tensor>& targets,
                                               std::shared_ptr<Tensor>& result,
                                               float epsilon) {
    // TODO: Implement when binary cross entropy pipeline is ready
    std::cerr << "Binary cross entropy forward GPU implementation not yet ready" << std::endl;
    return false;
}

bool LossOpsGPU::binary_cross_entropy_backward(const std::shared_ptr<Tensor>& predictions,
                                                const std::shared_ptr<Tensor>& targets,
                                                std::shared_ptr<Tensor>& gradient,
                                                float epsilon) {
    // TODO: Implement when binary cross entropy pipeline is ready
    std::cerr << "Binary cross entropy backward GPU implementation not yet ready" << std::endl;
    return false;
}

} // namespace dlvk
