#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/compute/compute_pipeline.h"
#include "dlvk/core/vulkan_device.h"
#include <iostream>
#include <cstring>

namespace dlvk {

TensorOps::TensorOps(std::shared_ptr<VulkanDevice> device) : m_device(device) {}

TensorOps::~TensorOps() {
    if (m_fence != VK_NULL_HANDLE) {
        vkDestroyFence(m_device->get_device(), m_fence, nullptr);
    }
    
    if (m_command_buffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device->get_device(), m_device->get_command_pool(), 1, &m_command_buffer);
    }
}

bool TensorOps::initialize() {
    if (!allocate_command_buffer()) {
        std::cerr << "Failed to allocate command buffer for tensor operations" << std::endl;
        return false;
    }
    
    // Create fence for synchronization
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    
    VkResult result = vkCreateFence(m_device->get_device(), &fence_info, nullptr, &m_fence);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create fence" << std::endl;
        return false;
    }
    
    if (!create_pipelines()) {
        std::cerr << "Failed to create compute pipelines" << std::endl;
        return false;
    }
    
    return true;
}

bool TensorOps::add(const Tensor& a, const Tensor& b, Tensor& result) {
    if (!validate_element_wise_operation(a, b, result)) {
        return false;
    }
    
    if (!m_add_pipeline) {
        std::cerr << "Add pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_add_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_add_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_add_pipeline->update_descriptor_set(0, 2, result.buffer());
    
    // Bind pipeline and dispatch
    m_add_pipeline->bind(cmd);
    
    // Push constants for size
    uint32_t size = static_cast<uint32_t>(a.size());
    m_add_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    // Calculate dispatch size (256 threads per workgroup)
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_add_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::add_broadcast(const Tensor& a, const Tensor& b, Tensor& result) {
    // Simple broadcast addition - for now, handle the case of adding 1D bias to 2D matrix
    // a: [batch_size, features], b: [features], result: [batch_size, features]
    
    if (a.shape() != result.shape()) {
        std::cerr << "Input A and result tensors must have same shape for broadcast addition" << std::endl;
        return false;
    }
    
    if (a.shape().size() == 2 && b.shape().size() == 1) {
        // Matrix + vector broadcast (common for bias addition)
        if (a.shape()[1] != b.shape()[0]) {
            std::cerr << "Incompatible shapes for broadcast addition" << std::endl;
            return false;
        }
        
        // For now, implement on CPU side - TODO: Create GPU kernel for broadcast
        std::vector<float> a_data(a.size());
        std::vector<float> b_data(b.size());
        std::vector<float> result_data(result.size());
        
        a.download_data(a_data.data());
        b.download_data(b_data.data());
        
        size_t batch_size = a.shape()[0];
        size_t features = a.shape()[1];
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < features; ++j) {
                result_data[i * features + j] = a_data[i * features + j] + b_data[j];
            }
        }
        
        result.upload_data(result_data.data());
        return true;
    } else {
        // Fallback to regular addition if shapes are compatible
        return add(a, b, result);
    }
}

bool TensorOps::multiply(const Tensor& a, const Tensor& b, Tensor& result) {
    if (!validate_element_wise_operation(a, b, result)) {
        return false;
    }
    
    if (!m_multiply_pipeline) {
        std::cerr << "Multiply pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_multiply_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_multiply_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_multiply_pipeline->update_descriptor_set(0, 2, result.buffer());
    
    // Bind pipeline and dispatch
    m_multiply_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(a.size());
    m_multiply_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_multiply_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::matrix_multiply(const Tensor& a, const Tensor& b, Tensor& result) {
    if (!validate_matrix_multiply(a, b, result)) {
        return false;
    }
    
    if (!m_matmul_pipeline) {
        std::cerr << "Matrix multiply pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_matmul_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_matmul_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_matmul_pipeline->update_descriptor_set(0, 2, result.buffer());
    
    // Bind pipeline
    m_matmul_pipeline->bind(cmd);
    
    // Push constants for matrix dimensions
    struct MatMulConstants {
        uint32_t M, N, P;
    } constants;
    
    constants.M = static_cast<uint32_t>(a.shape()[0]);  // rows of A
    constants.N = static_cast<uint32_t>(a.shape()[1]);  // cols of A, rows of B
    constants.P = static_cast<uint32_t>(b.shape()[1]);  // cols of B
    
    m_matmul_pipeline->push_constants(cmd, &constants, sizeof(constants));
    
    // Dispatch with 16x16 workgroups
    uint32_t workgroup_x = (constants.M + 15) / 16;
    uint32_t workgroup_y = (constants.P + 15) / 16;
    
    m_matmul_pipeline->dispatch(cmd, workgroup_x, workgroup_y);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::relu(const Tensor& input, Tensor& result) {
    if (input.shape() != result.shape() || input.dtype() != result.dtype()) {
        std::cerr << "Input and result tensors must have same shape and dtype" << std::endl;
        return false;
    }
    
    if (!m_relu_pipeline) {
        std::cerr << "ReLU pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_relu_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_relu_pipeline->update_descriptor_set(0, 1, result.buffer());
    
    // Bind pipeline and dispatch
    m_relu_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(input.size());
    m_relu_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_relu_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::sigmoid(const Tensor& input, Tensor& result) {
    if (input.shape() != result.shape() || input.dtype() != result.dtype()) {
        std::cerr << "Input and result tensors must have same shape and dtype" << std::endl;
        return false;
    }
    
    if (!m_sigmoid_pipeline) {
        std::cerr << "Sigmoid pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_sigmoid_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_sigmoid_pipeline->update_descriptor_set(0, 1, result.buffer());
    
    // Bind pipeline and dispatch
    m_sigmoid_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(input.size());
    m_sigmoid_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_sigmoid_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::tanh_activation(const Tensor& input, Tensor& result) {
    if (input.shape() != result.shape() || input.dtype() != result.dtype()) {
        std::cerr << "Input and result tensors must have same shape and dtype" << std::endl;
        return false;
    }
    
    if (!m_tanh_pipeline) {
        std::cerr << "Tanh pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_tanh_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_tanh_pipeline->update_descriptor_set(0, 1, result.buffer());
    
    // Bind pipeline and dispatch
    m_tanh_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(input.size());
    m_tanh_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_tanh_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::subtract(const Tensor& a, const Tensor& b, Tensor& result) {
    if (!validate_element_wise_operation(a, b, result)) {
        return false;
    }
    
    if (!m_subtract_pipeline) {
        std::cerr << "Subtract pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_subtract_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_subtract_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_subtract_pipeline->update_descriptor_set(0, 2, result.buffer());
    
    // Bind pipeline and dispatch
    m_subtract_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(a.size());
    m_subtract_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_subtract_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::divide(const Tensor& a, const Tensor& b, Tensor& result) {
    if (!validate_element_wise_operation(a, b, result)) {
        return false;
    }
    
    if (!m_divide_pipeline) {
        std::cerr << "Divide pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_divide_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_divide_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_divide_pipeline->update_descriptor_set(0, 2, result.buffer());
    
    // Bind pipeline and dispatch
    m_divide_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(a.size());
    m_divide_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_divide_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::transpose(const Tensor& input, Tensor& result) {
    if (input.shape().size() != 2 || result.shape().size() != 2) {
        std::cerr << "Transpose currently only supports 2D tensors" << std::endl;
        return false;
    }
    
    if (input.shape()[0] != result.shape()[1] || input.shape()[1] != result.shape()[0]) {
        std::cerr << "Result tensor has invalid shape for transpose" << std::endl;
        return false;
    }
    
    if (!m_transpose_pipeline) {
        std::cerr << "Transpose pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_transpose_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_transpose_pipeline->update_descriptor_set(0, 1, result.buffer());
    
    // Bind pipeline
    m_transpose_pipeline->bind(cmd);
    
    // Push constants for matrix dimensions
    struct TransposeConstants {
        uint32_t rows, cols;
    } constants;
    
    constants.rows = static_cast<uint32_t>(input.shape()[0]);
    constants.cols = static_cast<uint32_t>(input.shape()[1]);
    
    m_transpose_pipeline->push_constants(cmd, &constants, sizeof(constants));
    
    // Dispatch with 16x16 workgroups
    uint32_t workgroup_x = (constants.rows + 15) / 16;
    uint32_t workgroup_y = (constants.cols + 15) / 16;
    
    m_transpose_pipeline->dispatch(cmd, workgroup_x, workgroup_y);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::sum(const Tensor& input, Tensor& result, int axis) {
    if (axis == 0 && input.shape().size() == 2) {
        // Use specialized axis-0 reduction for 2D tensors
        return sum_axis0(input, result);
    }
    
    if (axis != -1) {
        std::cerr << "Only axis=0 and axis=-1 reductions implemented" << std::endl;
        return false;
    }
    
    // For now, implement simple total sum (axis=-1)
    if (result.size() != 1) {
        std::cerr << "Result tensor must have size 1 for total sum" << std::endl;
        return false;
    }
    
    if (!m_reduce_sum_pipeline) {
        std::cerr << "Reduce sum pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_reduce_sum_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_reduce_sum_pipeline->update_descriptor_set(0, 1, result.buffer());
    
    // Bind pipeline
    m_reduce_sum_pipeline->bind(cmd);
    
    // Push constants
    struct ReduceConstants {
        uint32_t input_size, output_size, reduction_size;
    } constants;
    
    constants.input_size = static_cast<uint32_t>(input.size());
    constants.output_size = 1;
    constants.reduction_size = constants.input_size;
    
    m_reduce_sum_pipeline->push_constants(cmd, &constants, sizeof(constants));
    
    // Dispatch
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (constants.input_size + workgroup_size - 1) / workgroup_size;
    
    m_reduce_sum_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::sum_axis0(const Tensor& input, Tensor& result) {
    // Sum along axis 0 (batch dimension)
    // Input: [batch_size, features] -> Output: [features]
    
    if (input.shape().size() != 2) {
        std::cerr << "sum_axis0 requires 2D input tensor" << std::endl;
        return false;
    }
    
    if (result.shape().size() != 1 || result.shape()[0] != input.shape()[1]) {
        std::cerr << "Result shape must be [features] for axis-0 reduction" << std::endl;
        return false;
    }
    
    if (!m_reduce_sum_axis0_pipeline) {
        std::cerr << "Sum axis-0 pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_reduce_sum_axis0_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_reduce_sum_axis0_pipeline->update_descriptor_set(0, 1, result.buffer());
    
    m_reduce_sum_axis0_pipeline->bind(cmd);
    
    struct PushConstants {
        uint32_t batch_size;
        uint32_t features;
        uint32_t total_size;
    } push_data;
    
    push_data.batch_size = static_cast<uint32_t>(input.shape()[0]);
    push_data.features = static_cast<uint32_t>(input.shape()[1]);
    push_data.total_size = static_cast<uint32_t>(input.size());
    
    m_reduce_sum_axis0_pipeline->push_constants(cmd, &push_data, sizeof(push_data));
    
    uint32_t workgroup_size = 64;
    uint32_t num_workgroups = (push_data.features + workgroup_size - 1) / workgroup_size;
    
    m_reduce_sum_axis0_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::mean(const Tensor& input, Tensor& result, int axis) {
    // Implement mean as sum / size
    if (!sum(input, result, axis)) {
        return false;
    }
    
    // Divide by size
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // For now, manually divide by size on CPU side
    std::vector<float> mean_data(result.size());
    result.download_data(mean_data.data());
    
    float size_factor = static_cast<float>(input.size());
    for (auto& val : mean_data) {
        val /= size_factor;
    }
    
    result.upload_data(mean_data.data());
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::max(const Tensor& input, Tensor& result, int axis) {
    // Simple implementation: find max across all elements
    if (axis != -1) {
        std::cerr << "Axis-specific max not yet implemented, using global max" << std::endl;
    }
    
    // Ensure result tensor is scalar (single element)
    if (result.size() != 1) {
        std::cerr << "Result tensor must be scalar for max operation" << std::endl;
        return false;
    }
    
    // Download input data, find max on CPU for now
    std::vector<float> input_data(input.size());
    input.download_data(input_data.data());
    
    // Find maximum value
    float max_val = input_data[0];
    for (size_t i = 1; i < input_data.size(); ++i) {
        if (input_data[i] > max_val) {
            max_val = input_data[i];
        }
    }
    
    // Upload result
    result.upload_data(&max_val);
    return true;
}

bool TensorOps::min(const Tensor& input, Tensor& result, int axis) {
    // Simple implementation: find min across all elements
    if (axis != -1) {
        std::cerr << "Axis-specific min not yet implemented, using global min" << std::endl;
    }
    
    // Ensure result tensor is scalar (single element)
    if (result.size() != 1) {
        std::cerr << "Result tensor must be scalar for min operation" << std::endl;
        return false;
    }
    
    // Download input data, find min on CPU for now
    std::vector<float> input_data(input.size());
    input.download_data(input_data.data());
    
    // Find minimum value
    float min_val = input_data[0];
    for (size_t i = 1; i < input_data.size(); ++i) {
        if (input_data[i] < min_val) {
            min_val = input_data[i];
        }
    }
    
    // Upload result
    result.upload_data(&min_val);
    return true;
}

bool TensorOps::softmax(const Tensor& input, Tensor& result) {
    if (input.shape() != result.shape()) {
        std::cerr << "Input and result tensors must have same shape for softmax" << std::endl;
        return false;
    }
    
    if (input.shape().size() != 2) {
        std::cerr << "Softmax currently only supports 2D tensors (batch_size, features)" << std::endl;
        return false;
    }
    
    if (!m_softmax_pipeline) {
        std::cerr << "Softmax pipeline not initialized" << std::endl;
        return false;
    }
    
    // Create temporary buffer for max values and sums
    size_t batch_size = input.shape()[0];
    size_t feature_size = input.shape()[1];
    
    VkBuffer temp_buffer;
    VkDeviceMemory temp_memory;
    
    VkResult res = m_device->create_buffer(
        batch_size * 2 * sizeof(float),  // Space for max values and sums
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        temp_buffer,
        temp_memory
    );
    
    if (res != VK_SUCCESS) {
        std::cerr << "Failed to create temporary buffer for softmax" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    // Update descriptor sets
    m_softmax_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_softmax_pipeline->update_descriptor_set(0, 1, result.buffer());
    m_softmax_pipeline->update_descriptor_set(0, 2, temp_buffer);
    
    // Bind pipeline
    m_softmax_pipeline->bind(cmd);
    
    struct SoftmaxConstants {
        uint32_t batch_size, feature_size, pass;
    } constants;
    
    constants.batch_size = static_cast<uint32_t>(batch_size);
    constants.feature_size = static_cast<uint32_t>(feature_size);
    
    uint32_t total_elements = constants.batch_size * constants.feature_size;
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
    
    // Three-pass softmax for numerical stability
    for (uint32_t pass = 0; pass < 3; ++pass) {
        constants.pass = pass;
        m_softmax_pipeline->push_constants(cmd, &constants, sizeof(constants));
        m_softmax_pipeline->dispatch(cmd, num_workgroups);
        
        // Add memory barrier between passes
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    
    end_single_time_commands(cmd);
    
    // Cleanup temporary buffer
    m_device->destroy_buffer(temp_buffer, temp_memory);
    
    return true;
}

bool TensorOps::fill(Tensor& tensor, float value) {
    // TODO: Implement fill operation
    std::cerr << "Fill operation not yet implemented" << std::endl;
    return false;
}

bool TensorOps::copy(const Tensor& source, Tensor& destination) {
    if (source.shape() != destination.shape() || source.dtype() != destination.dtype()) {
        std::cerr << "Source and destination tensors must have same shape and dtype" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    VkBufferCopy copy_region{};
    copy_region.size = source.size() * source.element_size();
    
    vkCmdCopyBuffer(cmd, source.buffer(), destination.buffer(), 1, &copy_region);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::create_pipelines() {
    std::cout << "Creating compute pipelines..." << std::endl;
    
    // Create simple descriptor set layout for 3 storage buffers (most operations)
    std::vector<VkDescriptorSetLayoutBinding> three_buffer_bindings(3);
    for (int i = 0; i < 3; ++i) {
        three_buffer_bindings[i].binding = i;
        three_buffer_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        three_buffer_bindings[i].descriptorCount = 1;
        three_buffer_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        three_buffer_bindings[i].pImmutableSamplers = nullptr;
    }
    
    // Create descriptor set layout for 2 storage buffers (activation functions)
    std::vector<VkDescriptorSetLayoutBinding> two_buffer_bindings(2);
    for (int i = 0; i < 2; ++i) {
        two_buffer_bindings[i].binding = i;
        two_buffer_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        two_buffer_bindings[i].descriptorCount = 1;
        two_buffer_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        two_buffer_bindings[i].pImmutableSamplers = nullptr;
    }
    
    // Setup push constant range
    PushConstantRange push_range;
    push_range.offset = 0;
    push_range.size = sizeof(uint32_t) * 4;  // Enough for most operations
    push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    // Try to create pipelines - if any fail, continue with others for debugging
    int success_count = 0;
    
    // Add pipeline
    m_add_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_add_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        m_add_pipeline->set_push_constant_range(push_range);
        if (m_add_pipeline->create_from_file("build/shaders/tensor_add.comp.spv")) {
            if (m_add_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Add pipeline created successfully" << std::endl;
                success_count++;
            } else {
                std::cout << "✗ Add pipeline descriptor allocation failed" << std::endl;
                m_add_pipeline.reset();
            }
        } else {
            std::cout << "✗ Add pipeline shader loading failed" << std::endl;
            m_add_pipeline.reset();
        }
    } else {
        std::cout << "✗ Add pipeline descriptor layout creation failed" << std::endl;
        m_add_pipeline.reset();
    }
    
    // Multiply pipeline
    m_multiply_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_multiply_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        m_multiply_pipeline->set_push_constant_range(push_range);
        if (m_multiply_pipeline->create_from_file("build/shaders/tensor_multiply.comp.spv")) {
            if (m_multiply_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Multiply pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_multiply_pipeline.reset();
            }
        } else {
            m_multiply_pipeline.reset();
        }
    } else {
        m_multiply_pipeline.reset();
    }
    
    // Matrix multiply pipeline
    m_matmul_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_matmul_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        PushConstantRange matmul_push_range;
        matmul_push_range.offset = 0;
        matmul_push_range.size = sizeof(uint32_t) * 3;  // M, N, P
        matmul_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_matmul_pipeline->set_push_constant_range(matmul_push_range);
        
        if (m_matmul_pipeline->create_from_file("build/shaders/matrix_multiply.comp.spv")) {
            if (m_matmul_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Matrix multiply pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_matmul_pipeline.reset();
            }
        } else {
            m_matmul_pipeline.reset();
        }
    } else {
        m_matmul_pipeline.reset();
    }
    
    // ReLU pipeline (2 buffers)
    m_relu_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_relu_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        PushConstantRange activation_push_range;
        activation_push_range.offset = 0;
        activation_push_range.size = sizeof(uint32_t);
        activation_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_relu_pipeline->set_push_constant_range(activation_push_range);
        
        if (m_relu_pipeline->create_from_file("build/shaders/relu.comp.spv")) {
            if (m_relu_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ ReLU pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_relu_pipeline.reset();
            }
        } else {
            m_relu_pipeline.reset();
        }
    } else {
        m_relu_pipeline.reset();
    }
    
    // Create Subtract pipeline
    m_subtract_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_subtract_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        m_subtract_pipeline->set_push_constant_range(push_range);
        if (m_subtract_pipeline->create_from_file("build/shaders/tensor_subtract.comp.spv")) {
            if (m_subtract_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Subtract pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_subtract_pipeline.reset();
            }
        } else {
            m_subtract_pipeline.reset();
        }
    } else {
        m_subtract_pipeline.reset();
    }
    
    // Create Divide pipeline
    m_divide_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_divide_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        m_divide_pipeline->set_push_constant_range(push_range);
        if (m_divide_pipeline->create_from_file("build/shaders/tensor_divide.comp.spv")) {
            if (m_divide_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Divide pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_divide_pipeline.reset();
            }
        } else {
            m_divide_pipeline.reset();
        }
    } else {
        m_divide_pipeline.reset();
    }
    
    // Create Sigmoid pipeline  
    m_sigmoid_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_sigmoid_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        m_sigmoid_pipeline->set_push_constant_range(push_range);
        if (m_sigmoid_pipeline->create_from_file("build/shaders/sigmoid.comp.spv")) {
            if (m_sigmoid_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Sigmoid pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_sigmoid_pipeline.reset();
            }
        } else {
            m_sigmoid_pipeline.reset();
        }
    } else {
        m_sigmoid_pipeline.reset();
    }
    
    // Create Reduce Sum pipeline
    m_reduce_sum_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_reduce_sum_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        // Push constants for reduction parameters
        PushConstantRange reduce_push_range;
        reduce_push_range.offset = 0;
        reduce_push_range.size = sizeof(uint32_t) * 3;  // input_size, output_size, reduction_size
        reduce_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_reduce_sum_pipeline->set_push_constant_range(reduce_push_range);
        
        if (m_reduce_sum_pipeline->create_from_file("build/shaders/reduce_sum.comp.spv")) {
            if (m_reduce_sum_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Reduce Sum pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_reduce_sum_pipeline.reset();
            }
        } else {
            m_reduce_sum_pipeline.reset();
        }
    } else {
        m_reduce_sum_pipeline.reset();
    }
    
    // Create Tanh pipeline
    m_tanh_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_tanh_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        PushConstantRange activation_push_range;
        activation_push_range.offset = 0;
        activation_push_range.size = sizeof(uint32_t);
        activation_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_tanh_pipeline->set_push_constant_range(activation_push_range);
        
        if (m_tanh_pipeline->create_from_file("build/shaders/tanh.comp.spv")) {
            if (m_tanh_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Tanh pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_tanh_pipeline.reset();
            }
        } else {
            m_tanh_pipeline.reset();
        }
    } else {
        m_tanh_pipeline.reset();
    }
    
    // Create Transpose pipeline
    m_transpose_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_transpose_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        // Push constants for matrix dimensions
        PushConstantRange transpose_push_range;
        transpose_push_range.offset = 0;
        transpose_push_range.size = sizeof(uint32_t) * 2;  // rows, cols
        transpose_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_transpose_pipeline->set_push_constant_range(transpose_push_range);
        
        if (m_transpose_pipeline->create_from_file("build/shaders/transpose.comp.spv")) {
            if (m_transpose_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Transpose pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_transpose_pipeline.reset();
            }
        } else {
            m_transpose_pipeline.reset();
        }
    } else {
        m_transpose_pipeline.reset();
    }
    
    // Create Softmax pipeline
    m_softmax_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_softmax_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {  // 3 buffers for softmax (input, output, temp)
        // Push constants for softmax parameters
        PushConstantRange softmax_push_range;
        softmax_push_range.offset = 0;
        softmax_push_range.size = sizeof(uint32_t) * 3;  // batch_size, feature_size, pass
        softmax_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_softmax_pipeline->set_push_constant_range(softmax_push_range);
        
        if (m_softmax_pipeline->create_from_file("build/shaders/softmax.comp.spv")) {
            if (m_softmax_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Softmax pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_softmax_pipeline.reset();
            }
        } else {
            m_softmax_pipeline.reset();
        }
    } else {
        m_softmax_pipeline.reset();
    }
    
    // Backward pass pipelines
    
    // ReLU backward pipeline (4 buffers: input, unused, grad_output, grad_input)
    std::vector<VkDescriptorSetLayoutBinding> four_buffer_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    
    m_relu_backward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_relu_backward_pipeline->create_descriptor_set_layout(four_buffer_bindings)) {
        PushConstantRange backward_push_range;
        backward_push_range.offset = 0;
        backward_push_range.size = sizeof(uint32_t);
        backward_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_relu_backward_pipeline->set_push_constant_range(backward_push_range);
        
        if (m_relu_backward_pipeline->create_from_file("build/shaders/relu_backward.comp.spv")) {
            if (m_relu_backward_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ ReLU backward pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_relu_backward_pipeline.reset();
            }
        } else {
            m_relu_backward_pipeline.reset();
        }
    } else {
        m_relu_backward_pipeline.reset();
    }
    
    // Sigmoid backward pipeline (3 buffers: output, grad_output, grad_input)
    std::vector<VkDescriptorSetLayoutBinding> backward_three_buffer_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    
    PushConstantRange backward_push_range;
    backward_push_range.offset = 0;
    backward_push_range.size = sizeof(uint32_t);
    backward_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    m_sigmoid_backward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_sigmoid_backward_pipeline->create_descriptor_set_layout(backward_three_buffer_bindings)) {
        m_sigmoid_backward_pipeline->set_push_constant_range(backward_push_range);
        
        if (m_sigmoid_backward_pipeline->create_from_file("build/shaders/sigmoid_backward.comp.spv")) {
            if (m_sigmoid_backward_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Sigmoid backward pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_sigmoid_backward_pipeline.reset();
            }
        } else {
            m_sigmoid_backward_pipeline.reset();
        }
    } else {
        m_sigmoid_backward_pipeline.reset();
    }
    
    // Tanh backward pipeline
    m_tanh_backward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_tanh_backward_pipeline->create_descriptor_set_layout(backward_three_buffer_bindings)) {
        m_tanh_backward_pipeline->set_push_constant_range(backward_push_range);
        
        if (m_tanh_backward_pipeline->create_from_file("build/shaders/tanh_backward.comp.spv")) {
            if (m_tanh_backward_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Tanh backward pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_tanh_backward_pipeline.reset();
            }
        } else {
            m_tanh_backward_pipeline.reset();
        }
    } else {
        m_tanh_backward_pipeline.reset();
    }
    
    // Reduce sum axis-0 pipeline (2 buffers: input, output) 
    m_reduce_sum_axis0_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_reduce_sum_axis0_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        // Push constants for batch_size, features, total_size
        PushConstantRange axis0_push_range;
        axis0_push_range.offset = 0;
        axis0_push_range.size = sizeof(uint32_t) * 3;
        axis0_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_reduce_sum_axis0_pipeline->set_push_constant_range(axis0_push_range);
        
        if (m_reduce_sum_axis0_pipeline->create_from_file("build/shaders/reduce_sum_axis0.comp.spv")) {
            if (m_reduce_sum_axis0_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Reduce sum axis-0 pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_reduce_sum_axis0_pipeline.reset();
            }
        } else {
            m_reduce_sum_axis0_pipeline.reset();
        }
    } else {
        m_reduce_sum_axis0_pipeline.reset();
    }
    
    // Create Conv2D pipeline
    std::vector<VkDescriptorSetLayoutBinding> conv2d_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weights
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // bias
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // output
    };
    dlvk::PushConstantRange conv2d_push_range = {0, sizeof(uint32_t) * 13, VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_conv2d_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_conv2d_pipeline->create_descriptor_set_layout(conv2d_bindings)) {
        m_conv2d_pipeline->set_push_constant_range(conv2d_push_range);
        if (m_conv2d_pipeline->create_from_file("/home/mostafizur/DLVK/build/shaders/conv2d.comp.spv")) {
            if (m_conv2d_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Conv2D pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_conv2d_pipeline.reset();
            }
        } else {
            m_conv2d_pipeline.reset();
        }
    } else {
        m_conv2d_pipeline.reset();
    }
    
    // Create MaxPool2D pipeline
    std::vector<VkDescriptorSetLayoutBinding> maxpool_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // indices
    };
    dlvk::PushConstantRange maxpool_push_range = {0, sizeof(uint32_t) * 12, VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_maxpool2d_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_maxpool2d_pipeline->create_descriptor_set_layout(maxpool_bindings)) {
        m_maxpool2d_pipeline->set_push_constant_range(maxpool_push_range);
        if (m_maxpool2d_pipeline->create_from_file("/home/mostafizur/DLVK/build/shaders/maxpool2d.comp.spv")) {
            if (m_maxpool2d_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ MaxPool2D pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_maxpool2d_pipeline.reset();
            }
        } else {
            m_maxpool2d_pipeline.reset();
        }
    } else {
        m_maxpool2d_pipeline.reset();
    }
    
    // Create BatchNorm pipeline
    std::vector<VkDescriptorSetLayoutBinding> batchnorm_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // gamma
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // beta
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // running_mean
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // running_var
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // saved_mean
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // saved_var
    };
    dlvk::PushConstantRange batchnorm_push_range = {0, sizeof(uint32_t) * 4 + sizeof(float) * 2, VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_batch_norm_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_batch_norm_pipeline->create_descriptor_set_layout(batchnorm_bindings)) {
        m_batch_norm_pipeline->set_push_constant_range(batchnorm_push_range);
        if (m_batch_norm_pipeline->create_from_file("/home/mostafizur/DLVK/build/shaders/batch_norm.comp.spv")) {
            if (m_batch_norm_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ BatchNorm pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_batch_norm_pipeline.reset();
            }
        } else {
            m_batch_norm_pipeline.reset();
        }
    } else {
        m_batch_norm_pipeline.reset();
    }
    
    // Create Dropout pipeline
    std::vector<VkDescriptorSetLayoutBinding> dropout_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // mask
    };
    dlvk::PushConstantRange dropout_push_range = {0, sizeof(uint32_t) * 3 + sizeof(float) * 2, VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_dropout_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_dropout_pipeline->create_descriptor_set_layout(dropout_bindings)) {
        m_dropout_pipeline->set_push_constant_range(dropout_push_range);
        if (m_dropout_pipeline->create_from_file("/home/mostafizur/DLVK/build/shaders/dropout.comp.spv")) {
            if (m_dropout_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Dropout pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_dropout_pipeline.reset();
            }
        } else {
            m_dropout_pipeline.reset();
        }
    } else {
        m_dropout_pipeline.reset();
    }

    // Create avgpool2d pipeline
    VkDescriptorSetLayoutBinding avgpool_bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
    };
    dlvk::PushConstantRange avgpool_push_range = {0, sizeof(uint32_t) * 12, VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_avgpool2d_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_avgpool2d_pipeline->create_descriptor_set_layout(std::vector<VkDescriptorSetLayoutBinding>(avgpool_bindings, avgpool_bindings + 2))) {
        m_avgpool2d_pipeline->set_push_constant_range(avgpool_push_range);
        if (m_avgpool2d_pipeline->create_from_file("/home/mostafizur/DLVK/build/shaders/avgpool2d.comp.spv")) {
            if (m_avgpool2d_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ AvgPool2D pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_avgpool2d_pipeline.reset();
            }
        } else {
            m_avgpool2d_pipeline.reset();
        }
    } else {
        m_avgpool2d_pipeline.reset();
    }

    // Create batch_norm_backward pipeline
    VkDescriptorSetLayoutBinding batchnorm_backward_bindings[8] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // grad_output
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // gamma
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // saved_mean
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // saved_var
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // grad_input
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // grad_gamma
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // grad_beta
    };
    dlvk::PushConstantRange batchnorm_backward_push_range = {0, sizeof(uint32_t) * 3 + sizeof(float), VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_batch_norm_backward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_batch_norm_backward_pipeline->create_descriptor_set_layout(std::vector<VkDescriptorSetLayoutBinding>(batchnorm_backward_bindings, batchnorm_backward_bindings + 8))) {
        m_batch_norm_backward_pipeline->set_push_constant_range(batchnorm_backward_push_range);
        if (m_batch_norm_backward_pipeline->create_from_file("/home/mostafizur/DLVK/build/shaders/batch_norm_backward.comp.spv")) {
            if (m_batch_norm_backward_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ BatchNorm backward pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_batch_norm_backward_pipeline.reset();
            }
        } else {
            m_batch_norm_backward_pipeline.reset();
        }
    } else {
        m_batch_norm_backward_pipeline.reset();
    }

    // Create dropout_backward pipeline
    VkDescriptorSetLayoutBinding dropout_backward_bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // grad_output
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // mask
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // grad_input
    };
    dlvk::PushConstantRange dropout_backward_push_range = {0, sizeof(uint32_t) + sizeof(float), VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_dropout_backward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_dropout_backward_pipeline->create_descriptor_set_layout(std::vector<VkDescriptorSetLayoutBinding>(dropout_backward_bindings, dropout_backward_bindings + 3))) {
        m_dropout_backward_pipeline->set_push_constant_range(dropout_backward_push_range);
        if (m_dropout_backward_pipeline->create_from_file("/home/mostafizur/DLVK/build/shaders/dropout_backward.comp.spv")) {
            if (m_dropout_backward_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Dropout backward pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_dropout_backward_pipeline.reset();
            }
        } else {
            m_dropout_backward_pipeline.reset();
        }
    } else {
        m_dropout_backward_pipeline.reset();
    }

    std::cout << "Pipeline creation summary: " << success_count << " pipelines created" << std::endl;
    
    // Return true if we have at least some working pipelines
    return success_count > 0;
}

bool TensorOps::allocate_command_buffer() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = m_device->get_command_pool();
    alloc_info.commandBufferCount = 1;
    
    VkResult result = vkAllocateCommandBuffers(m_device->get_device(), &alloc_info, &m_command_buffer);
    return result == VK_SUCCESS;
}

VkCommandBuffer TensorOps::begin_single_time_commands() {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(m_command_buffer, &begin_info);
    
    return m_command_buffer;
}

void TensorOps::end_single_time_commands(VkCommandBuffer cmd_buffer) {
    vkEndCommandBuffer(cmd_buffer);
    
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;
    
    // Reset fence
    vkResetFences(m_device->get_device(), 1, &m_fence);
    
    // Submit and wait
    vkQueueSubmit(m_device->get_compute_queue(), 1, &submit_info, m_fence);
    vkWaitForFences(m_device->get_device(), 1, &m_fence, VK_TRUE, UINT64_MAX);
}

bool TensorOps::validate_element_wise_operation(const Tensor& a, const Tensor& b, const Tensor& result) {
    if (a.shape() != b.shape()) {
        std::cerr << "Input tensors must have same shape for element-wise operation" << std::endl;
        return false;
    }
    
    if (a.shape() != result.shape()) {
        std::cerr << "Result tensor must have same shape as input tensors" << std::endl;
        return false;
    }
    
    if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
        std::cerr << "All tensors must have same data type" << std::endl;
        return false;
    }
    
    return true;
}

// CNN Operations Implementation

bool TensorOps::conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output,
                       size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w) {
    if (!m_conv2d_pipeline) {
        std::cerr << "Conv2D pipeline not initialized" << std::endl;
        return false;
    }
    
    // Validate shapes
    const auto& input_shape = input.shape();
    const auto& weight_shape = weights.shape();
    const auto& output_shape = output.shape();
    
    if (input_shape.size() != 4 || weight_shape.size() != 4 || output_shape.size() != 4) {
        std::cerr << "Conv2D requires 4D tensors [batch, channels, height, width]" << std::endl;
        return false;
    }
    
    // Extract dimensions
    uint32_t batch_size = static_cast<uint32_t>(input_shape[0]);
    uint32_t in_channels = static_cast<uint32_t>(input_shape[1]);
    uint32_t input_height = static_cast<uint32_t>(input_shape[2]);
    uint32_t input_width = static_cast<uint32_t>(input_shape[3]);
    
    uint32_t out_channels = static_cast<uint32_t>(weight_shape[0]);
    uint32_t kernel_height = static_cast<uint32_t>(weight_shape[2]);
    uint32_t kernel_width = static_cast<uint32_t>(weight_shape[3]);
    
    uint32_t output_height = static_cast<uint32_t>(output_shape[2]);
    uint32_t output_width = static_cast<uint32_t>(output_shape[3]);
    
    // Setup push constants
    struct PushConstants {
        uint32_t batch_size;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t input_height;
        uint32_t input_width;
        uint32_t output_height;
        uint32_t output_width;
        uint32_t kernel_height;
        uint32_t kernel_width;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
    } push_constants;
    
    push_constants.batch_size = batch_size;
    push_constants.in_channels = in_channels;
    push_constants.out_channels = out_channels;
    push_constants.input_height = input_height;
    push_constants.input_width = input_width;
    push_constants.output_height = output_height;
    push_constants.output_width = output_width;
    push_constants.kernel_height = kernel_height;
    push_constants.kernel_width = kernel_width;
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    
    // Update descriptor sets
    m_conv2d_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_conv2d_pipeline->update_descriptor_set(0, 1, weights.buffer());
    m_conv2d_pipeline->update_descriptor_set(0, 2, bias.buffer());
    m_conv2d_pipeline->update_descriptor_set(0, 3, output.buffer());
    
    // Dispatch compute
    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_conv2d_pipeline->bind(cmd);
    m_conv2d_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    
    uint32_t group_count_x = (output_width + 7) / 8;
    uint32_t group_count_y = (output_height + 7) / 8;
    uint32_t group_count_z = batch_size;
    
    m_conv2d_pipeline->dispatch(cmd, group_count_x, group_count_y, group_count_z);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::maxpool2d(const Tensor& input, Tensor& output, Tensor& indices,
                          size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                          size_t padding_h, size_t padding_w) {
    if (!m_maxpool2d_pipeline) {
        std::cerr << "MaxPool2D pipeline not initialized" << std::endl;
        return false;
    }
    
    const auto& input_shape = input.shape();
    const auto& output_shape = output.shape();
    
    if (input_shape.size() != 4 || output_shape.size() != 4) {
        std::cerr << "MaxPool2D requires 4D tensors [batch, channels, height, width]" << std::endl;
        return false;
    }
    
    uint32_t batch_size = static_cast<uint32_t>(input_shape[0]);
    uint32_t channels = static_cast<uint32_t>(input_shape[1]);
    uint32_t input_height = static_cast<uint32_t>(input_shape[2]);
    uint32_t input_width = static_cast<uint32_t>(input_shape[3]);
    uint32_t output_height = static_cast<uint32_t>(output_shape[2]);
    uint32_t output_width = static_cast<uint32_t>(output_shape[3]);
    
    struct PushConstants {
        uint32_t batch_size;
        uint32_t channels;
        uint32_t input_height;
        uint32_t input_width;
        uint32_t output_height;
        uint32_t output_width;
        uint32_t pool_height;
        uint32_t pool_width;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
    } push_constants;
    
    push_constants.batch_size = batch_size;
    push_constants.channels = channels;
    push_constants.input_height = input_height;
    push_constants.input_width = input_width;
    push_constants.output_height = output_height;
    push_constants.output_width = output_width;
    push_constants.pool_height = static_cast<uint32_t>(pool_h);
    push_constants.pool_width = static_cast<uint32_t>(pool_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    
    // Update descriptor sets
    m_maxpool2d_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_maxpool2d_pipeline->update_descriptor_set(0, 1, output.buffer());
    m_maxpool2d_pipeline->update_descriptor_set(0, 2, indices.buffer());
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_maxpool2d_pipeline->bind(cmd);
    m_maxpool2d_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    
    uint32_t group_count_x = (output_width + 7) / 8;
    uint32_t group_count_y = (output_height + 7) / 8;
    uint32_t group_count_z = batch_size * channels;
    
    m_maxpool2d_pipeline->dispatch(cmd, group_count_x, group_count_y, group_count_z);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::batch_norm(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                           Tensor& running_mean, Tensor& running_var,
                           Tensor& output, Tensor& saved_mean, Tensor& saved_var,
                           float momentum, float epsilon, bool training) {
    if (!m_batch_norm_pipeline) {
        std::cerr << "BatchNorm pipeline not initialized" << std::endl;
        return false;
    }
    
    uint32_t batch_size = static_cast<uint32_t>(input.shape()[0]);
    uint32_t num_features = static_cast<uint32_t>(gamma.size());
    uint32_t total_elements = static_cast<uint32_t>(input.size());
    
    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_features;
        uint32_t total_elements;
        float momentum;
        float epsilon;
        uint32_t training;
    } push_constants;
    
    push_constants.batch_size = batch_size;
    push_constants.num_features = num_features;
    push_constants.total_elements = total_elements;
    push_constants.momentum = momentum;
    push_constants.epsilon = epsilon;
    push_constants.training = training ? 1 : 0;
    
    // Update descriptor sets
    m_batch_norm_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_batch_norm_pipeline->update_descriptor_set(0, 1, gamma.buffer());
    m_batch_norm_pipeline->update_descriptor_set(0, 2, beta.buffer());
    m_batch_norm_pipeline->update_descriptor_set(0, 3, running_mean.buffer());
    m_batch_norm_pipeline->update_descriptor_set(0, 4, running_var.buffer());
    m_batch_norm_pipeline->update_descriptor_set(0, 5, output.buffer());
    m_batch_norm_pipeline->update_descriptor_set(0, 6, saved_mean.buffer());
    m_batch_norm_pipeline->update_descriptor_set(0, 7, saved_var.buffer());
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_batch_norm_pipeline->bind(cmd);
    m_batch_norm_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    
    m_batch_norm_pipeline->dispatch(cmd, num_features, 1, 1);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::dropout(const Tensor& input, Tensor& output, Tensor& mask,
                        float dropout_rate, bool training, uint32_t seed) {
    if (!m_dropout_pipeline) {
        std::cerr << "Dropout pipeline not initialized" << std::endl;
        return false;
    }
    
    uint32_t total_elements = static_cast<uint32_t>(input.size());
    float scale_factor = training ? (1.0f / (1.0f - dropout_rate)) : 1.0f;
    
    struct PushConstants {
        uint32_t total_elements;
        float dropout_rate;
        float scale_factor;
        uint32_t training;
        uint32_t seed;
    } push_constants;
    
    push_constants.total_elements = total_elements;
    push_constants.dropout_rate = dropout_rate;
    push_constants.scale_factor = scale_factor;
    push_constants.training = training ? 1 : 0;
    push_constants.seed = seed;
    
    // Update descriptor sets
    m_dropout_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_dropout_pipeline->update_descriptor_set(0, 1, output.buffer());
    m_dropout_pipeline->update_descriptor_set(0, 2, mask.buffer());
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_dropout_pipeline->bind(cmd);
    m_dropout_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    
    uint32_t group_count = (total_elements + 255) / 256;
    m_dropout_pipeline->dispatch(cmd, group_count, 1, 1);
    
    end_single_time_commands(cmd);
    
    return true;
}

// Placeholder implementations for other operations
bool TensorOps::conv2d_backward_input(const Tensor& grad_output, const Tensor& weights, Tensor& grad_input,
                                      size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w) {
    // TODO: Implement conv2d backward input pass
    std::cerr << "Conv2D backward input not yet implemented" << std::endl;
    return false;
}

bool TensorOps::conv2d_backward_weight(const Tensor& input, const Tensor& grad_output, 
                                       Tensor& grad_weights, Tensor& grad_bias,
                                       size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w) {
    // TODO: Implement conv2d backward weight pass
    std::cerr << "Conv2D backward weight not yet implemented" << std::endl;
    return false;
}

bool TensorOps::maxpool2d_backward(const Tensor& grad_output, const Tensor& indices, Tensor& grad_input) {
    // TODO: Implement maxpool2d backward pass
    std::cerr << "MaxPool2D backward not yet implemented" << std::endl;
    return false;
}

bool TensorOps::avgpool2d(const Tensor& input, Tensor& output,
                          size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                          size_t padding_h, size_t padding_w) {
    // TODO: Implement avgpool2d
    std::cerr << "AvgPool2D not yet implemented" << std::endl;
    return false;
}

bool TensorOps::avgpool2d_backward(const Tensor& grad_output, Tensor& grad_input,
                                   size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                                   size_t padding_h, size_t padding_w) {
    // TODO: Implement avgpool2d backward pass
    std::cerr << "AvgPool2D backward not yet implemented" << std::endl;
    return false;
}

bool TensorOps::batch_norm_backward(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& gamma, const Tensor& saved_mean, const Tensor& saved_var,
                                    Tensor& grad_input, Tensor& grad_gamma, Tensor& grad_beta,
                                    float epsilon) {
    // TODO: Implement batch norm backward pass
    std::cerr << "BatchNorm backward not yet implemented" << std::endl;
    return false;
}

bool TensorOps::dropout_backward(const Tensor& grad_output, const Tensor& mask, Tensor& grad_input,
                                 float dropout_rate) {
    // TODO: Implement dropout backward pass
    std::cerr << "Dropout backward not yet implemented" << std::endl;
    return false;
}

bool TensorOps::validate_matrix_multiply(const Tensor& a, const Tensor& b, const Tensor& result) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        std::cerr << "Matrix multiplication requires 2D tensors" << std::endl;
        return false;
    }
    
    if (a.shape()[1] != b.shape()[0]) {
        std::cerr << "Invalid dimensions for matrix multiplication" << std::endl;
        return false;
    }
    
    if (result.shape().size() != 2 || 
        result.shape()[0] != a.shape()[0] || 
        result.shape()[1] != b.shape()[1]) {
        std::cerr << "Result tensor has invalid shape for matrix multiplication" << std::endl;
        return false;
    }
    
    return true;
}

// Backward pass implementations
bool TensorOps::relu_backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
    if (!validate_element_wise_operation(input, grad_output, grad_input)) {
        return false;
    }
    
    if (!m_relu_backward_pipeline) {
        std::cerr << "ReLU backward pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_relu_backward_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_relu_backward_pipeline->update_descriptor_set(0, 1, VK_NULL_HANDLE); // Not used in this shader
    m_relu_backward_pipeline->update_descriptor_set(0, 2, grad_output.buffer());
    m_relu_backward_pipeline->update_descriptor_set(0, 3, grad_input.buffer());
    
    m_relu_backward_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(input.size());
    m_relu_backward_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 64;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_relu_backward_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::sigmoid_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input) {
    if (!validate_element_wise_operation(output, grad_output, grad_input)) {
        return false;
    }
    
    if (!m_sigmoid_backward_pipeline) {
        std::cerr << "Sigmoid backward pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_sigmoid_backward_pipeline->update_descriptor_set(0, 0, output.buffer());
    m_sigmoid_backward_pipeline->update_descriptor_set(0, 1, grad_output.buffer());
    m_sigmoid_backward_pipeline->update_descriptor_set(0, 2, grad_input.buffer());
    
    m_sigmoid_backward_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(output.size());
    m_sigmoid_backward_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 64;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_sigmoid_backward_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::tanh_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input) {
    if (!validate_element_wise_operation(output, grad_output, grad_input)) {
        return false;
    }
    
    if (!m_tanh_backward_pipeline) {
        std::cerr << "Tanh backward pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_tanh_backward_pipeline->update_descriptor_set(0, 0, output.buffer());
    m_tanh_backward_pipeline->update_descriptor_set(0, 1, grad_output.buffer());
    m_tanh_backward_pipeline->update_descriptor_set(0, 2, grad_input.buffer());
    
    m_tanh_backward_pipeline->bind(cmd);
    
    uint32_t size = static_cast<uint32_t>(output.size());
    m_tanh_backward_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    
    uint32_t workgroup_size = 64;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_tanh_backward_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

} // namespace dlvk
