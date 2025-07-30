#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/compute/compute_pipeline.h"
#include "dlvk/core/vulkan_device.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
    

    m_add_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_add_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_add_pipeline->update_descriptor_set(0, 2, result.buffer());
    

    m_add_pipeline->bind(cmd);
    

    uint32_t size = static_cast<uint32_t>(a.size());
    m_add_pipeline->push_constants(cmd, &size, sizeof(uint32_t));
    

    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (size + workgroup_size - 1) / workgroup_size;
    
    m_add_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::add_broadcast(const Tensor& a, const Tensor& b, Tensor& result) {


    
    if (a.shape() != result.shape()) {
        std::cerr << "Input A and result tensors must have same shape for broadcast addition" << std::endl;
        return false;
    }
    
    if (a.shape().size() == 2 && b.shape().size() == 1) {

        if (a.shape()[1] != b.shape()[0]) {
            std::cerr << "Incompatible shapes for broadcast addition" << std::endl;
            return false;
        }
        

        if (m_broadcast_add_pipeline) {

            m_broadcast_add_pipeline->update_descriptor_set(0, 0, a.buffer());
            m_broadcast_add_pipeline->update_descriptor_set(0, 1, b.buffer());
            m_broadcast_add_pipeline->update_descriptor_set(0, 2, result.buffer());
            

            struct PushConstants {
                uint32_t batch_size;
                uint32_t features;
                uint32_t total_size;
            } push_constants;
            
            push_constants.batch_size = static_cast<uint32_t>(a.shape()[0]);
            push_constants.features = static_cast<uint32_t>(a.shape()[1]);
            push_constants.total_size = static_cast<uint32_t>(a.size());
            

            VkCommandBuffer cmd = begin_single_time_commands();
            m_broadcast_add_pipeline->bind(cmd);
            m_broadcast_add_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
            
            uint32_t dispatch_x = (a.size() + 255) / 256;
            vkCmdDispatch(cmd, dispatch_x, 1, 1);
            
            end_single_time_commands(cmd);
            
            return true;
        }
        

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
    

    m_multiply_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_multiply_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_multiply_pipeline->update_descriptor_set(0, 2, result.buffer());
    

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
    

    m_matmul_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_matmul_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_matmul_pipeline->update_descriptor_set(0, 2, result.buffer());
    

    m_matmul_pipeline->bind(cmd);
    

    struct MatMulConstants {
        uint32_t M, N, P;
    } constants;
    
    constants.M = static_cast<uint32_t>(a.shape()[0]);  // rows of A
    constants.N = static_cast<uint32_t>(a.shape()[1]);  // cols of A, rows of B
    constants.P = static_cast<uint32_t>(b.shape()[1]);  // cols of B
    
    m_matmul_pipeline->push_constants(cmd, &constants, sizeof(constants));
    

    uint32_t workgroup_x = (constants.M + 15) / 16;
    uint32_t workgroup_y = (constants.P + 15) / 16;
    
    m_matmul_pipeline->dispatch(cmd, workgroup_x, workgroup_y);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::batch_matrix_multiply(const Tensor& a, const Tensor& b, Tensor& result) {

    if (a.dtype() != DataType::FLOAT32 || b.dtype() != DataType::FLOAT32 || result.dtype() != DataType::FLOAT32) {
        std::cerr << "All tensors must be float32 for batch matrix multiplication" << std::endl;
        return false;
    }
    
    if (a.shape().size() != 3 || b.shape().size() != 3 || result.shape().size() != 3) {
        std::cerr << "All tensors must be 3D for batch matrix multiplication [batch, rows, cols]" << std::endl;
        return false;
    }
    

    if (a.shape()[0] != b.shape()[0] || a.shape()[0] != result.shape()[0]) {
        std::cerr << "Batch dimensions must match for batch matrix multiplication" << std::endl;
        return false;
    }
    

    if (a.shape()[2] != b.shape()[1]) {
        std::cerr << "Inner dimensions must match: a.shape[2] != b.shape[1]" << std::endl;
        return false;
    }
    
    if (result.shape()[1] != a.shape()[1] || result.shape()[2] != b.shape()[2]) {
        std::cerr << "Result dimensions incorrect for batch matrix multiplication" << std::endl;
        return false;
    }
    
    if (!m_batch_matmul_pipeline) {
        std::cerr << "Batch matrix multiply pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    

    m_batch_matmul_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_batch_matmul_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_batch_matmul_pipeline->update_descriptor_set(0, 2, result.buffer());
    

    m_batch_matmul_pipeline->bind(cmd);
    

    struct BatchMatMulConstants {
        uint32_t batch_size, M, N, K, stride_a, stride_b, stride_c;
    } constants;
    
    constants.batch_size = static_cast<uint32_t>(a.shape()[0]);  // batch size
    constants.M = static_cast<uint32_t>(a.shape()[1]);          // rows of A
    constants.N = static_cast<uint32_t>(b.shape()[2]);          // cols of B
    constants.K = static_cast<uint32_t>(a.shape()[2]);          // cols of A, rows of B
    constants.stride_a = constants.M * constants.K;             // Elements per A matrix
    constants.stride_b = constants.K * constants.N;             // Elements per B matrix  
    constants.stride_c = constants.M * constants.N;             // Elements per C matrix
    
    m_batch_matmul_pipeline->push_constants(cmd, &constants, sizeof(constants));
    

    uint32_t workgroup_x = (constants.M + 15) / 16;
    uint32_t workgroup_y = (constants.N + 15) / 16;
    uint32_t workgroup_z = constants.batch_size;  // One workgroup per batch
    
    m_batch_matmul_pipeline->dispatch(cmd, workgroup_x, workgroup_y, workgroup_z);
    
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
    

    m_relu_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_relu_pipeline->update_descriptor_set(0, 1, result.buffer());
    

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
    

    m_sigmoid_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_sigmoid_pipeline->update_descriptor_set(0, 1, result.buffer());
    

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
    

    m_tanh_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_tanh_pipeline->update_descriptor_set(0, 1, result.buffer());
    

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
    

    m_subtract_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_subtract_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_subtract_pipeline->update_descriptor_set(0, 2, result.buffer());
    

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
    

    m_divide_pipeline->update_descriptor_set(0, 0, a.buffer());
    m_divide_pipeline->update_descriptor_set(0, 1, b.buffer());
    m_divide_pipeline->update_descriptor_set(0, 2, result.buffer());
    

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
    

    m_transpose_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_transpose_pipeline->update_descriptor_set(0, 1, result.buffer());
    

    m_transpose_pipeline->bind(cmd);
    

    struct TransposeConstants {
        uint32_t rows, cols;
    } constants;
    
    constants.rows = static_cast<uint32_t>(input.shape()[0]);
    constants.cols = static_cast<uint32_t>(input.shape()[1]);
    
    m_transpose_pipeline->push_constants(cmd, &constants, sizeof(constants));
    

    uint32_t workgroup_x = (constants.rows + 15) / 16;
    uint32_t workgroup_y = (constants.cols + 15) / 16;
    
    m_transpose_pipeline->dispatch(cmd, workgroup_x, workgroup_y);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::sum(const Tensor& input, Tensor& result, int axis) {
    if (axis == 0 && input.shape().size() == 2) {

        return sum_axis0(input, result);
    }
    
    if (axis != -1) {
        std::cerr << "Only axis=0 and axis=-1 reductions implemented" << std::endl;
        return false;
    }
    

    if (result.size() != 1) {
        std::cerr << "Result tensor must have size 1 for total sum" << std::endl;
        return false;
    }
    
    if (!m_reduce_sum_pipeline) {
        std::cerr << "Reduce sum pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    

    m_reduce_sum_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_reduce_sum_pipeline->update_descriptor_set(0, 1, result.buffer());
    

    m_reduce_sum_pipeline->bind(cmd);
    

    struct ReduceConstants {
        uint32_t input_size, output_size, reduction_size;
    } constants;
    
    constants.input_size = static_cast<uint32_t>(input.size());
    constants.output_size = 1;
    constants.reduction_size = constants.input_size;
    
    m_reduce_sum_pipeline->push_constants(cmd, &constants, sizeof(constants));
    

    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (constants.input_size + workgroup_size - 1) / workgroup_size;
    
    m_reduce_sum_pipeline->dispatch(cmd, num_workgroups);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::sum_axis0(const Tensor& input, Tensor& result) {


    
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

    if (!sum(input, result, axis)) {
        return false;
    }
    

    VkCommandBuffer cmd = begin_single_time_commands();
    

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

    if (axis != -1) {
        std::cerr << "Axis-specific max not yet implemented, using global max" << std::endl;
    }
    

    if (result.size() != 1) {
        std::cerr << "Result tensor must be scalar for max operation" << std::endl;
        return false;
    }
    

    std::vector<float> input_data(input.size());
    input.download_data(input_data.data());
    

    float max_val = input_data[0];
    for (size_t i = 1; i < input_data.size(); ++i) {
        if (input_data[i] > max_val) {
            max_val = input_data[i];
        }
    }
    

    result.upload_data(&max_val);
    return true;
}

bool TensorOps::min(const Tensor& input, Tensor& result, int axis) {

    if (axis != -1) {
        std::cerr << "Axis-specific min not yet implemented, using global min" << std::endl;
    }
    

    if (result.size() != 1) {
        std::cerr << "Result tensor must be scalar for min operation" << std::endl;
        return false;
    }
    

    std::vector<float> input_data(input.size());
    input.download_data(input_data.data());
    

    float min_val = input_data[0];
    for (size_t i = 1; i < input_data.size(); ++i) {
        if (input_data[i] < min_val) {
            min_val = input_data[i];
        }
    }
    

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
    

    m_softmax_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_softmax_pipeline->update_descriptor_set(0, 1, result.buffer());
    m_softmax_pipeline->update_descriptor_set(0, 2, temp_buffer);
    

    m_softmax_pipeline->bind(cmd);
    
    struct SoftmaxConstants {
        uint32_t batch_size, feature_size, pass;
    } constants;
    
    constants.batch_size = static_cast<uint32_t>(batch_size);
    constants.feature_size = static_cast<uint32_t>(feature_size);
    
    uint32_t total_elements = constants.batch_size * constants.feature_size;
    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
    

    for (uint32_t pass = 0; pass < 3; ++pass) {
        constants.pass = pass;
        m_softmax_pipeline->push_constants(cmd, &constants, sizeof(constants));
        m_softmax_pipeline->dispatch(cmd, num_workgroups);
        

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    
    end_single_time_commands(cmd);
    

    m_device->destroy_buffer(temp_buffer, temp_memory);
    
    return true;
}

bool TensorOps::fill(Tensor& tensor, float value) {
    if (!tensor.buffer()) {
        std::cerr << "Tensor buffer is null" << std::endl;
        return false;
    }


    VkDevice device = m_device->get_device();
    void* mapped_data;
    VkResult result = vkMapMemory(device, tensor.memory(), 0, VK_WHOLE_SIZE, 0, &mapped_data);
    
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map tensor memory for fill operation" << std::endl;
        return false;
    }

    size_t element_count = tensor.size();
    float* float_data = static_cast<float*>(mapped_data);
    
    for (size_t i = 0; i < element_count; ++i) {
        float_data[i] = value;
    }

    vkUnmapMemory(device, tensor.memory());
    return true;
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
    

    std::vector<VkDescriptorSetLayoutBinding> three_buffer_bindings(3);
    for (int i = 0; i < 3; ++i) {
        three_buffer_bindings[i].binding = i;
        three_buffer_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        three_buffer_bindings[i].descriptorCount = 1;
        three_buffer_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        three_buffer_bindings[i].pImmutableSamplers = nullptr;
    }
    

    std::vector<VkDescriptorSetLayoutBinding> two_buffer_bindings(2);
    for (int i = 0; i < 2; ++i) {
        two_buffer_bindings[i].binding = i;
        two_buffer_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        two_buffer_bindings[i].descriptorCount = 1;
        two_buffer_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        two_buffer_bindings[i].pImmutableSamplers = nullptr;
    }
    

    PushConstantRange push_range;
    push_range.offset = 0;
    push_range.size = sizeof(uint32_t) * 4;  // Enough for most operations
    push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
    

    int success_count = 0;
    

    m_add_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_add_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        m_add_pipeline->set_push_constant_range(push_range);
        if (m_add_pipeline->create_from_file("shaders/tensor_add.comp.spv")) {
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
    

    m_multiply_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_multiply_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        m_multiply_pipeline->set_push_constant_range(push_range);
        if (m_multiply_pipeline->create_from_file("shaders/tensor_multiply.comp.spv")) {
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
    

    m_matmul_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_matmul_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        PushConstantRange matmul_push_range;
        matmul_push_range.offset = 0;
        matmul_push_range.size = sizeof(uint32_t) * 3;  // M, N, P
        matmul_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_matmul_pipeline->set_push_constant_range(matmul_push_range);
        
        if (m_matmul_pipeline->create_from_file("shaders/matrix_multiply.comp.spv")) {
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
    

    m_batch_matmul_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_batch_matmul_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        PushConstantRange batch_matmul_push_range;
        batch_matmul_push_range.offset = 0;
        batch_matmul_push_range.size = sizeof(uint32_t) * 7;  // batch_size, M, N, K, stride_a, stride_b, stride_c
        batch_matmul_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_batch_matmul_pipeline->set_push_constant_range(batch_matmul_push_range);
        
        if (m_batch_matmul_pipeline->create_from_file("shaders/batch_matrix_multiply.comp.spv")) {
            if (m_batch_matmul_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Batch matrix multiply pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_batch_matmul_pipeline.reset();
            }
        } else {
            m_batch_matmul_pipeline.reset();
        }
    } else {
        m_batch_matmul_pipeline.reset();
    }
    

    m_relu_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_relu_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        PushConstantRange activation_push_range;
        activation_push_range.offset = 0;
        activation_push_range.size = sizeof(uint32_t);
        activation_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_relu_pipeline->set_push_constant_range(activation_push_range);
        
        if (m_relu_pipeline->create_from_file("shaders/relu.comp.spv")) {
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
    

    m_subtract_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_subtract_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        m_subtract_pipeline->set_push_constant_range(push_range);
        if (m_subtract_pipeline->create_from_file("shaders/tensor_subtract.comp.spv")) {
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
    

    m_divide_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_divide_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        m_divide_pipeline->set_push_constant_range(push_range);
        if (m_divide_pipeline->create_from_file("shaders/tensor_divide.comp.spv")) {
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
    

    m_sigmoid_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_sigmoid_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        m_sigmoid_pipeline->set_push_constant_range(push_range);
        if (m_sigmoid_pipeline->create_from_file("shaders/sigmoid.comp.spv")) {
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
    

    m_reduce_sum_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_reduce_sum_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {

        PushConstantRange reduce_push_range;
        reduce_push_range.offset = 0;
        reduce_push_range.size = sizeof(uint32_t) * 3;  // input_size, output_size, reduction_size
        reduce_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_reduce_sum_pipeline->set_push_constant_range(reduce_push_range);
        
        if (m_reduce_sum_pipeline->create_from_file("shaders/reduce_sum.comp.spv")) {
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
    

    m_tanh_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_tanh_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        PushConstantRange activation_push_range;
        activation_push_range.offset = 0;
        activation_push_range.size = sizeof(uint32_t);
        activation_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_tanh_pipeline->set_push_constant_range(activation_push_range);
        
        if (m_tanh_pipeline->create_from_file("shaders/tanh.comp.spv")) {
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
    

    m_transpose_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_transpose_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {

        PushConstantRange transpose_push_range;
        transpose_push_range.offset = 0;
        transpose_push_range.size = sizeof(uint32_t) * 2;  // rows, cols
        transpose_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_transpose_pipeline->set_push_constant_range(transpose_push_range);
        
        if (m_transpose_pipeline->create_from_file("shaders/transpose.comp.spv")) {
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
    

    m_softmax_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_softmax_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {  // 3 buffers for softmax (input, output, temp)

        PushConstantRange softmax_push_range;
        softmax_push_range.offset = 0;
        softmax_push_range.size = sizeof(uint32_t) * 3;  // batch_size, feature_size, pass
        softmax_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_softmax_pipeline->set_push_constant_range(softmax_push_range);
        
        if (m_softmax_pipeline->create_from_file("shaders/softmax.comp.spv")) {
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
        
        if (m_relu_backward_pipeline->create_from_file("shaders/relu_backward.comp.spv")) {
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
        
        if (m_sigmoid_backward_pipeline->create_from_file("shaders/sigmoid_backward.comp.spv")) {
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
    

    m_tanh_backward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_tanh_backward_pipeline->create_descriptor_set_layout(backward_three_buffer_bindings)) {
        m_tanh_backward_pipeline->set_push_constant_range(backward_push_range);
        
        if (m_tanh_backward_pipeline->create_from_file("shaders/tanh_backward.comp.spv")) {
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
    

    m_reduce_sum_axis0_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_reduce_sum_axis0_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {

        PushConstantRange axis0_push_range;
        axis0_push_range.offset = 0;
        axis0_push_range.size = sizeof(uint32_t) * 3;
        axis0_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_reduce_sum_axis0_pipeline->set_push_constant_range(axis0_push_range);
        
        if (m_reduce_sum_axis0_pipeline->create_from_file("shaders/reduce_sum_axis0.comp.spv")) {
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
        if (m_conv2d_pipeline->create_from_file("shaders/conv2d.comp.spv")) {
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
    

    std::vector<VkDescriptorSetLayoutBinding> maxpool_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // indices
    };
    dlvk::PushConstantRange maxpool_push_range = {0, sizeof(uint32_t) * 12, VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_maxpool2d_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_maxpool2d_pipeline->create_descriptor_set_layout(maxpool_bindings)) {
        m_maxpool2d_pipeline->set_push_constant_range(maxpool_push_range);
        if (m_maxpool2d_pipeline->create_from_file("shaders/maxpool2d.comp.spv")) {
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
        if (m_batch_norm_pipeline->create_from_file("shaders/batch_norm.comp.spv")) {
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
    

    std::vector<VkDescriptorSetLayoutBinding> dropout_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // mask
    };
    dlvk::PushConstantRange dropout_push_range = {0, sizeof(uint32_t) * 3 + sizeof(float) * 2, VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_dropout_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_dropout_pipeline->create_descriptor_set_layout(dropout_bindings)) {
        m_dropout_pipeline->set_push_constant_range(dropout_push_range);
        if (m_dropout_pipeline->create_from_file("shaders/dropout.comp.spv")) {
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


    VkDescriptorSetLayoutBinding avgpool_bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output
    };
    dlvk::PushConstantRange avgpool_push_range = {0, sizeof(uint32_t) * 12, VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_avgpool2d_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_avgpool2d_pipeline->create_descriptor_set_layout(std::vector<VkDescriptorSetLayoutBinding>(avgpool_bindings, avgpool_bindings + 2))) {
        m_avgpool2d_pipeline->set_push_constant_range(avgpool_push_range);
        if (m_avgpool2d_pipeline->create_from_file("shaders/avgpool2d.comp.spv")) {
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
        if (m_batch_norm_backward_pipeline->create_from_file("shaders/batch_norm_backward.comp.spv")) {
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


    VkDescriptorSetLayoutBinding dropout_backward_bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // grad_output
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // mask
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // grad_input
    };
    dlvk::PushConstantRange dropout_backward_push_range = {0, sizeof(uint32_t) + sizeof(float), VK_SHADER_STAGE_COMPUTE_BIT};
    
    m_dropout_backward_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_dropout_backward_pipeline->create_descriptor_set_layout(std::vector<VkDescriptorSetLayoutBinding>(dropout_backward_bindings, dropout_backward_bindings + 3))) {
        m_dropout_backward_pipeline->set_push_constant_range(dropout_backward_push_range);
        if (m_dropout_backward_pipeline->create_from_file("shaders/dropout_backward.comp.spv")) {
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


    

    m_scalar_multiply_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_scalar_multiply_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        PushConstantRange scalar_push_range;
        scalar_push_range.offset = 0;
        scalar_push_range.size = sizeof(uint32_t) + sizeof(float);  // size + scalar
        scalar_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_scalar_multiply_pipeline->set_push_constant_range(scalar_push_range);
        
        if (m_scalar_multiply_pipeline->create_from_file("build/shaders/scalar_multiply.comp.spv")) {
            if (m_scalar_multiply_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Scalar Multiply pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_scalar_multiply_pipeline.reset();
            }
        } else {
            m_scalar_multiply_pipeline.reset();
        }
    } else {
        m_scalar_multiply_pipeline.reset();
    }


    m_broadcast_add_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_broadcast_add_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        PushConstantRange broadcast_push_range;
        broadcast_push_range.offset = 0;
        broadcast_push_range.size = sizeof(uint32_t) * 3;  // batch_size, features, total_size
        broadcast_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_broadcast_add_pipeline->set_push_constant_range(broadcast_push_range);
        
        if (m_broadcast_add_pipeline->create_from_file("build/shaders/broadcast_add.comp.spv")) {
            if (m_broadcast_add_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Broadcast Add pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_broadcast_add_pipeline.reset();
            }
        } else {
            m_broadcast_add_pipeline.reset();
        }
    } else {
        m_broadcast_add_pipeline.reset();
    }


    m_sqrt_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_sqrt_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        PushConstantRange sqrt_push_range;
        sqrt_push_range.offset = 0;
        sqrt_push_range.size = sizeof(uint32_t);  // size
        sqrt_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_sqrt_pipeline->set_push_constant_range(sqrt_push_range);
        
        if (m_sqrt_pipeline->create_from_file("build/shaders/sqrt.comp.spv")) {
            if (m_sqrt_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Sqrt pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_sqrt_pipeline.reset();
            }
        } else {
            m_sqrt_pipeline.reset();
        }
    } else {
        m_sqrt_pipeline.reset();
    }


    m_clamp_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_clamp_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        PushConstantRange clamp_push_range;
        clamp_push_range.offset = 0;
        clamp_push_range.size = sizeof(uint32_t) + sizeof(float) * 2;  // size + min_val + max_val
        clamp_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_clamp_pipeline->set_push_constant_range(clamp_push_range);
        
        if (m_clamp_pipeline->create_from_file("build/shaders/clamp.comp.spv")) {
            if (m_clamp_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Clamp pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_clamp_pipeline.reset();
            }
        } else {
            m_clamp_pipeline.reset();
        }
    } else {
        m_clamp_pipeline.reset();
    }


    m_adam_update_pipeline = std::make_unique<ComputePipeline>(m_device);
    

    std::vector<VkDescriptorSetLayoutBinding> adam_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // params
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // gradients
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // momentum
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // velocity
    };
    
    if (m_adam_update_pipeline->create_descriptor_set_layout(adam_bindings)) {
        
        PushConstantRange adam_push_range;
        adam_push_range.offset = 0;
        adam_push_range.size = sizeof(uint32_t) + sizeof(float) * 6;  // size + lr + beta1 + beta2 + epsilon + bias_corrections
        adam_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_adam_update_pipeline->set_push_constant_range(adam_push_range);
        
        if (m_adam_update_pipeline->create_from_file("build/shaders/adam_update.comp.spv")) {
            if (m_adam_update_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Adam update pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_adam_update_pipeline.reset();
            }
        } else {
            m_adam_update_pipeline.reset();
        }
    } else {
        m_adam_update_pipeline.reset();
    }


    

    m_embedding_lookup_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_embedding_lookup_pipeline->create_descriptor_set_layout(three_buffer_bindings)) {
        PushConstantRange embedding_push_range;
        embedding_push_range.offset = 0;
        embedding_push_range.size = sizeof(uint32_t) * 4;  // batch_size, sequence_length, embedding_dim, num_embeddings
        embedding_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_embedding_lookup_pipeline->set_push_constant_range(embedding_push_range);
        
        if (m_embedding_lookup_pipeline->create_from_file("build/shaders/embedding_lookup.comp.spv")) {
            if (m_embedding_lookup_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Embedding lookup pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_embedding_lookup_pipeline.reset();
            }
        } else {
            m_embedding_lookup_pipeline.reset();
        }
    } else {
        m_embedding_lookup_pipeline.reset();
    }


    m_layer_norm_pipeline = std::make_unique<ComputePipeline>(m_device);
    std::vector<VkDescriptorSetLayoutBinding> layer_norm_bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // weight
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // bias
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // output
    };
    
    if (m_layer_norm_pipeline->create_descriptor_set_layout(layer_norm_bindings)) {
        PushConstantRange layer_norm_push_range;
        layer_norm_push_range.offset = 0;
        layer_norm_push_range.size = sizeof(uint32_t) * 3 + sizeof(float);  // batch_size, seq_length, feature_dim, eps
        layer_norm_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_layer_norm_pipeline->set_push_constant_range(layer_norm_push_range);
        
        if (m_layer_norm_pipeline->create_from_file("build/shaders/layer_norm.comp.spv")) {
            if (m_layer_norm_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Layer norm pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_layer_norm_pipeline.reset();
            }
        } else {
            m_layer_norm_pipeline.reset();
        }
    } else {
        m_layer_norm_pipeline.reset();
    }


    m_multi_head_attention_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_multi_head_attention_pipeline) {

        std::vector<VkDescriptorSetLayoutBinding> bindings(4);
        for (int i = 0; i < 4; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }
        
        if (m_multi_head_attention_pipeline->create_descriptor_set_layout(bindings)) {

            PushConstantRange attention_push_range{};
            attention_push_range.offset = 0;
            attention_push_range.size = sizeof(uint32_t) * 5 + sizeof(float);  // batch_size, seq_length, num_heads, head_dim, total_elements, scale
            attention_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
            m_multi_head_attention_pipeline->set_push_constant_range(attention_push_range);
            
            if (m_multi_head_attention_pipeline->create_from_file("build/shaders/multi_head_attention.comp.spv")) {
                if (m_multi_head_attention_pipeline->allocate_descriptor_sets(1)) {
                    std::cout << "✓ Multi-head attention pipeline created successfully" << std::endl;
                    success_count++;
                } else {
                    m_multi_head_attention_pipeline.reset();
                }
            } else {
                m_multi_head_attention_pipeline.reset();
            }
        } else {
            m_multi_head_attention_pipeline.reset();
        }
    } else {
        m_multi_head_attention_pipeline.reset();
    }


    m_tensor_reshape_pipeline = std::make_unique<ComputePipeline>(m_device);
    if (m_tensor_reshape_pipeline->create_descriptor_set_layout(two_buffer_bindings)) {
        PushConstantRange reshape_push_range{};
        reshape_push_range.offset = 0;
        reshape_push_range.size = sizeof(uint32_t) * 10;  // total_elements, input_dims[4], output_dims[4], num_dims
        reshape_push_range.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
        m_tensor_reshape_pipeline->set_push_constant_range(reshape_push_range);
        
        if (m_tensor_reshape_pipeline->create_from_file("shaders/tensor_reshape.comp.spv")) {
            if (m_tensor_reshape_pipeline->allocate_descriptor_sets(1)) {
                std::cout << "✓ Tensor reshape pipeline created successfully" << std::endl;
                success_count++;
            } else {
                m_tensor_reshape_pipeline.reset();
            }
        } else {
            m_tensor_reshape_pipeline.reset();
        }
    } else {
        m_tensor_reshape_pipeline.reset();
    }

    std::cout << "Pipeline creation summary: " << success_count << " pipelines created" << std::endl;
    

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

    vkResetCommandBuffer(m_command_buffer, 0);
    
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
    

    vkResetFences(m_device->get_device(), 1, &m_fence);
    

    VkResult submit_result = vkQueueSubmit(m_device->get_compute_queue(), 1, &submit_info, m_fence);
    if (submit_result != VK_SUCCESS) {
        std::cerr << "Failed to submit command buffer: " << submit_result << std::endl;
        return;
    }
    
    VkResult fence_result = vkWaitForFences(m_device->get_device(), 1, &m_fence, VK_TRUE, UINT64_MAX);
    if (fence_result != VK_SUCCESS) {
        std::cerr << "Failed to wait for fence: " << fence_result << std::endl;
        return;
    }
    

    VkResult queue_result = vkQueueWaitIdle(m_device->get_compute_queue());
    if (queue_result != VK_SUCCESS) {
        std::cerr << "Failed to wait for queue idle: " << queue_result << std::endl;
        return;
    }
    

    VkResult device_result = vkDeviceWaitIdle(m_device->get_device());
    if (device_result != VK_SUCCESS) {
        std::cerr << "Failed to wait for device idle: " << device_result << std::endl;
        return;
    }
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



bool TensorOps::conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output,
                       size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w) {
    if (!m_conv2d_pipeline) {
        std::cerr << "Conv2D pipeline not initialized" << std::endl;
        return false;
    }
    

    const auto& input_shape = input.shape();
    const auto& weight_shape = weights.shape();
    const auto& output_shape = output.shape();
    
    if (input_shape.size() != 4 || weight_shape.size() != 4 || output_shape.size() != 4) {
        std::cerr << "Conv2D requires 4D tensors [batch, channels, height, width]" << std::endl;
        return false;
    }
    

    uint32_t batch_size = static_cast<uint32_t>(input_shape[0]);
    uint32_t in_channels = static_cast<uint32_t>(input_shape[1]);
    uint32_t input_height = static_cast<uint32_t>(input_shape[2]);
    uint32_t input_width = static_cast<uint32_t>(input_shape[3]);
    
    uint32_t out_channels = static_cast<uint32_t>(weight_shape[0]);
    uint32_t kernel_height = static_cast<uint32_t>(weight_shape[2]);
    uint32_t kernel_width = static_cast<uint32_t>(weight_shape[3]);
    
    uint32_t output_height = static_cast<uint32_t>(output_shape[2]);
    uint32_t output_width = static_cast<uint32_t>(output_shape[3]);
    

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
    

    m_conv2d_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_conv2d_pipeline->update_descriptor_set(0, 1, weights.buffer());
    m_conv2d_pipeline->update_descriptor_set(0, 2, bias.buffer());
    m_conv2d_pipeline->update_descriptor_set(0, 3, output.buffer());
    

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


bool TensorOps::conv2d_backward_input(const Tensor& grad_output, const Tensor& weights, Tensor& grad_input,
                                      size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w) {

    std::cerr << "Conv2D backward input not yet implemented" << std::endl;
    return false;
}

bool TensorOps::conv2d_backward_weight(const Tensor& input, const Tensor& grad_output, 
                                       Tensor& grad_weights, Tensor& grad_bias,
                                       size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w) {

    std::cerr << "Conv2D backward weight not yet implemented" << std::endl;
    return false;
}

bool TensorOps::maxpool2d_backward(const Tensor& grad_output, const Tensor& indices, Tensor& grad_input) {

    std::cerr << "MaxPool2D backward not yet implemented" << std::endl;
    return false;
}

bool TensorOps::avgpool2d(const Tensor& input, Tensor& output,
                          size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                          size_t padding_h, size_t padding_w) {

    std::cerr << "AvgPool2D not yet implemented" << std::endl;
    return false;
}

bool TensorOps::avgpool2d_backward(const Tensor& grad_output, Tensor& grad_input,
                                   size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                                   size_t padding_h, size_t padding_w) {

    std::cerr << "AvgPool2D backward not yet implemented" << std::endl;
    return false;
}

bool TensorOps::batch_norm_backward(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& gamma, const Tensor& saved_mean, const Tensor& saved_var,
                                    Tensor& grad_input, Tensor& grad_gamma, Tensor& grad_beta,
                                    float epsilon) {

    std::cerr << "BatchNorm backward not yet implemented" << std::endl;
    return false;
}

bool TensorOps::dropout_backward(const Tensor& grad_output, const Tensor& mask, Tensor& grad_input,
                                 float dropout_rate) {

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


bool TensorOps::scale(const Tensor& input, float scalar, Tensor& result) {

    auto scalar_tensor = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, m_device);
    scalar_tensor->upload_data(&scalar);
    
    return multiply(input, *scalar_tensor, result);
}

bool TensorOps::scalar_add(const Tensor& input, float scalar, Tensor& result) {

    auto scalar_tensor = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, m_device);
    scalar_tensor->upload_data(&scalar);
    
    return add(input, *scalar_tensor, result);
}

bool TensorOps::element_wise_multiply(const Tensor& a, const Tensor& b, Tensor& result) {
    return multiply(a, b, result);
}



bool TensorOps::scalar_multiply(const Tensor& input, float scalar, Tensor& result) {
    if (!m_scalar_multiply_pipeline) {
        return false; // Pipeline not available, caller should use CPU fallback
    }
    
    if (input.size() != result.size()) {
        std::cerr << "Scalar multiply: input and result tensor sizes don't match" << std::endl;
        return false;
    }
    

    m_scalar_multiply_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_scalar_multiply_pipeline->update_descriptor_set(0, 1, result.buffer());
    

    struct PushConstants {
        uint32_t size;
        float scalar;
    } push_constants;
    
    push_constants.size = static_cast<uint32_t>(input.size());
    push_constants.scalar = scalar;
    

    VkCommandBuffer cmd = begin_single_time_commands();
    m_scalar_multiply_pipeline->bind(cmd);
    m_scalar_multiply_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    
    uint32_t dispatch_x = (input.size() + 255) / 256;
    vkCmdDispatch(cmd, dispatch_x, 1, 1);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::clamp(const Tensor& input, float min_val, float max_val, Tensor& result) {
    if (!m_clamp_pipeline) {
        return false; // Pipeline not available, caller should use CPU fallback
    }
    
    if (input.size() != result.size()) {
        std::cerr << "Clamp: input and result tensor sizes don't match" << std::endl;
        return false;
    }
    

    m_clamp_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_clamp_pipeline->update_descriptor_set(0, 1, result.buffer());
    

    struct PushConstants {
        uint32_t size;
        float min_val;
        float max_val;
    } push_constants;
    
    push_constants.size = static_cast<uint32_t>(input.size());
    push_constants.min_val = min_val;
    push_constants.max_val = max_val;
    

    VkCommandBuffer cmd = begin_single_time_commands();
    m_clamp_pipeline->bind(cmd);
    m_clamp_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    
    uint32_t dispatch_x = (input.size() + 255) / 256;
    vkCmdDispatch(cmd, dispatch_x, 1, 1);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::element_wise_sqrt(const Tensor& input, Tensor& result) {

    if (m_sqrt_pipeline) {
        if (input.size() != result.size()) {
            std::cerr << "Sqrt: input and result tensor sizes don't match" << std::endl;
            return false;
        }
        

        m_sqrt_pipeline->update_descriptor_set(0, 0, input.buffer());
        m_sqrt_pipeline->update_descriptor_set(0, 1, result.buffer());
        

        uint32_t size = static_cast<uint32_t>(input.size());
        

        VkCommandBuffer cmd = begin_single_time_commands();
        m_sqrt_pipeline->bind(cmd);
        m_sqrt_pipeline->push_constants(cmd, &size, sizeof(size));
        
        uint32_t dispatch_x = (input.size() + 255) / 256;
        vkCmdDispatch(cmd, dispatch_x, 1, 1);
        
        end_single_time_commands(cmd);
        
        return true;
    }
    

    std::vector<float> input_data(input.size());
    std::vector<float> result_data(input.size());
    
    const_cast<Tensor&>(input).download_data(input_data.data());
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        result_data[i] = std::sqrt(input_data[i]);
    }
    
    result.upload_data(result_data.data());
    return true;
}

bool TensorOps::element_wise_square(const Tensor& input, Tensor& result) {
    return multiply(input, input, result);
}

bool TensorOps::adam_update(const Tensor& gradient, const Tensor& m, const Tensor& v, 
                           Tensor& param, Tensor& new_m, Tensor& new_v,
                           float lr, float beta1, float beta2, float epsilon) {

    if (gradient.shape() != m.shape() || gradient.shape() != v.shape() || 
        gradient.shape() != param.shape()) {
        std::cerr << "All tensors must have the same shape for Adam update" << std::endl;
        return false;
    }
    

    if (m_adam_update_pipeline && gradient.size() > 0) {
        VkCommandBuffer cmd = begin_single_time_commands();
        
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_adam_update_pipeline->get_pipeline());
        

        std::vector<VkDescriptorBufferInfo> buffer_infos = {
            {param.buffer(), 0, VK_WHOLE_SIZE},
            {gradient.buffer(), 0, VK_WHOLE_SIZE},
            {m.buffer(), 0, VK_WHOLE_SIZE},
            {v.buffer(), 0, VK_WHOLE_SIZE}
        };
        
        std::vector<VkWriteDescriptorSet> writes(4);
        for (size_t i = 0; i < 4; ++i) {
            writes[i] = {};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = m_adam_update_pipeline->get_descriptor_set(0);
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        
        vkUpdateDescriptorSets(m_device->get_device(), writes.size(), writes.data(), 0, nullptr);
        
        VkDescriptorSet descriptor_set = m_adam_update_pipeline->get_descriptor_set(0);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                               m_adam_update_pipeline->get_layout(), 0, 1, 
                               &descriptor_set, 0, nullptr);
        

        struct AdamPushConstants {
            uint32_t size;
            float lr;
            float beta1; 
            float beta2;
            float epsilon;
            float bias_correction1;  // Will be set by optimizer
            float bias_correction2;  // Will be set by optimizer
        } pc;
        
        pc.size = static_cast<uint32_t>(gradient.size());
        pc.lr = lr;
        pc.beta1 = beta1;
        pc.beta2 = beta2;
        pc.epsilon = epsilon;
        pc.bias_correction1 = 1.0f;  // Simplified for now
        pc.bias_correction2 = 1.0f;  // Simplified for now
        
        vkCmdPushConstants(cmd, m_adam_update_pipeline->get_layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        
        uint32_t dispatch_x = (gradient.size() + 255) / 256;
        vkCmdDispatch(cmd, dispatch_x, 1, 1);
        
        end_single_time_commands(cmd);
        

        copy(m, new_m);
        copy(v, new_v);
        
        return true;
    }
    

    std::cout << "Using CPU fallback for Adam update" << std::endl;
    std::vector<float> grad_data(gradient.size());
    std::vector<float> m_data(m.size());
    std::vector<float> v_data(v.size());
    std::vector<float> param_data(param.size());
    
    const_cast<Tensor&>(gradient).download_data(grad_data.data());
    const_cast<Tensor&>(m).download_data(m_data.data());
    const_cast<Tensor&>(v).download_data(v_data.data());
    param.download_data(param_data.data());
    
    for (size_t i = 0; i < grad_data.size(); ++i) {

        m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * grad_data[i];
        

        v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * grad_data[i] * grad_data[i];
        

        param_data[i] = param_data[i] - lr * m_data[i] / (std::sqrt(v_data[i]) + epsilon);
    }
    

    new_m.upload_data(m_data.data());
    new_v.upload_data(v_data.data());
    param.upload_data(param_data.data());
    
    return true;
}

bool TensorOps::gradient_clip_by_norm(const Tensor& gradient, float max_norm, Tensor& clipped_gradient) {
    if (gradient.shape() != clipped_gradient.shape() || gradient.dtype() != clipped_gradient.dtype()) {
        std::cerr << "Gradient and clipped_gradient tensors must have same shape and dtype" << std::endl;
        return false;
    }
    

    auto squared_grad = std::make_shared<Tensor>(gradient.shape(), gradient.dtype(), gradient.device());
    auto norm_tensor = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, gradient.device());
    
    if (!element_wise_square(gradient, *squared_grad)) {
        return false;
    }
    
    if (!sum(*squared_grad, *norm_tensor)) {
        return false;
    }
    

    std::vector<float> norm_data(1);
    norm_tensor->download_data(norm_data.data());
    float norm = std::sqrt(norm_data[0]);
    
    if (norm <= max_norm) {

        return copy(gradient, clipped_gradient);
    } else {

        float scale_factor = max_norm / norm;
        return scale(gradient, scale_factor, clipped_gradient);
    }
}

bool TensorOps::gradient_clip_by_value(const Tensor& gradient, float min_val, float max_val, Tensor& clipped_gradient) {
    if (gradient.shape() != clipped_gradient.shape() || gradient.dtype() != clipped_gradient.dtype()) {
        std::cerr << "Gradient and clipped_gradient tensors must have same shape and dtype" << std::endl;
        return false;
    }
    


    VkCommandBuffer cmd = begin_single_time_commands();
    

    if (!copy(gradient, clipped_gradient)) {
        return false;
    }
    


    std::vector<float> grad_data(gradient.size());
    clipped_gradient.download_data(grad_data.data());
    
    for (auto& val : grad_data) {
        val = std::max(min_val, std::min(max_val, val));
    }
    
    clipped_gradient.upload_data(grad_data.data());
    
    end_single_time_commands(cmd);
    return true;
}





bool TensorOps::embedding_lookup(Tensor& output, const Tensor& indices, const Tensor& embeddings) {

    const auto& indices_shape = indices.shape();
    const auto& embeddings_shape = embeddings.shape();
    const auto& output_shape = output.shape();
    
    if (indices_shape.size() != 2) {
        std::cerr << "Embedding indices must be 2D (batch_size, sequence_length)" << std::endl;
        return false;
    }
    
    if (embeddings_shape.size() != 2) {
        std::cerr << "Embeddings must be 2D (num_embeddings, embedding_dim)" << std::endl;
        return false;
    }
    
    if (output_shape.size() != 3) {
        std::cerr << "Output must be 3D (batch_size, sequence_length, embedding_dim)" << std::endl;
        return false;
    }
    

    size_t batch_size = indices_shape[0];
    size_t sequence_length = indices_shape[1];
    size_t num_embeddings = embeddings_shape[0];
    size_t embedding_dim = embeddings_shape[1];
    
    if (output_shape[0] != batch_size || output_shape[1] != sequence_length || output_shape[2] != embedding_dim) {
        std::cerr << "Output shape mismatch for embedding lookup" << std::endl;
        return false;
    }
    
    if (!m_embedding_lookup_pipeline) {
        std::cerr << "Embedding lookup pipeline not initialized" << std::endl;
        return false;
    }
    

    struct PushConstants {
        uint32_t batch_size;
        uint32_t sequence_length;
        uint32_t embedding_dim;
        uint32_t num_embeddings;
    } push_constants;
    
    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.sequence_length = static_cast<uint32_t>(sequence_length);
    push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
    push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);
    

    m_embedding_lookup_pipeline->update_descriptor_set(0, 0, output.buffer());
    m_embedding_lookup_pipeline->update_descriptor_set(0, 1, indices.buffer());
    m_embedding_lookup_pipeline->update_descriptor_set(0, 2, embeddings.buffer());
    

    VkCommandBuffer cmd = begin_single_time_commands();
    
    m_embedding_lookup_pipeline->bind(cmd);
    m_embedding_lookup_pipeline->push_constants(cmd, &push_constants, sizeof(push_constants));
    

    uint32_t total_elements = static_cast<uint32_t>(output.size());
    uint32_t workgroup_size = 64;
    uint32_t num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
    
    m_embedding_lookup_pipeline->dispatch(cmd, num_workgroups, 1, 1);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::layer_norm(Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, float eps) {
    if (!m_layer_norm_pipeline) {
        std::cerr << "Layer norm pipeline not initialized" << std::endl;
        return false;
    }
    
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto bias_shape = bias.shape();
    auto output_shape = output.shape();
    


    if (input_shape.size() != 3) {
        std::cerr << "Input must be 3D for layer norm" << std::endl;
        return false;
    }
    
    size_t batch_size = input_shape[0];
    size_t seq_length = input_shape[1]; 
    size_t feature_dim = input_shape[2];
    
    if (weight_shape.size() != 1 || weight_shape[0] != feature_dim ||
        bias_shape.size() != 1 || bias_shape[0] != feature_dim) {
        std::cerr << "Weight and bias must be 1D with size matching feature_dim" << std::endl;
        return false;
    }
    
    if (output_shape != input_shape) {
        std::cerr << "Output shape must match input shape" << std::endl;
        return false;
    }
    

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    

    std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}  // 4 buffers: input, weight, bias, output
    };
    
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    
    if (vkCreateDescriptorPool(m_device->get_device(), &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool for layer norm" << std::endl;
        return false;
    }
    

    VkDescriptorSetLayout layout = m_layer_norm_pipeline->get_descriptor_layout();
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;
    
    if (vkAllocateDescriptorSets(m_device->get_device(), &alloc_info, &descriptor_set) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device->get_device(), descriptor_pool, nullptr);
        std::cerr << "Failed to allocate descriptor set for layer norm" << std::endl;
        return false;
    }
    

    std::vector<VkWriteDescriptorSet> descriptor_writes(4);
    
    VkDescriptorBufferInfo input_buffer_info{};
    input_buffer_info.buffer = input.buffer();
    input_buffer_info.offset = 0;
    input_buffer_info.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo weight_buffer_info{};
    weight_buffer_info.buffer = weight.buffer();
    weight_buffer_info.offset = 0;
    weight_buffer_info.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo bias_buffer_info{};
    bias_buffer_info.buffer = bias.buffer();
    bias_buffer_info.offset = 0;
    bias_buffer_info.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo output_buffer_info{};
    output_buffer_info.buffer = output.buffer();
    output_buffer_info.offset = 0;
    output_buffer_info.range = VK_WHOLE_SIZE;
    
    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = descriptor_set;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &input_buffer_info;
    
    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].dstSet = descriptor_set;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].pBufferInfo = &weight_buffer_info;
    
    descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[2].dstSet = descriptor_set;
    descriptor_writes[2].dstBinding = 2;
    descriptor_writes[2].dstArrayElement = 0;
    descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[2].descriptorCount = 1;
    descriptor_writes[2].pBufferInfo = &bias_buffer_info;
    
    descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[3].dstSet = descriptor_set;
    descriptor_writes[3].dstBinding = 3;
    descriptor_writes[3].dstArrayElement = 0;
    descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[3].descriptorCount = 1;
    descriptor_writes[3].pBufferInfo = &output_buffer_info;
    
    vkUpdateDescriptorSets(m_device->get_device(), static_cast<uint32_t>(descriptor_writes.size()), 
                          descriptor_writes.data(), 0, nullptr);
    

    struct PushConstants {
        uint32_t batch_size;
        uint32_t seq_length;
        uint32_t feature_dim;
        float eps;
    } push_constants;
    
    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.seq_length = static_cast<uint32_t>(seq_length);
    push_constants.feature_dim = static_cast<uint32_t>(feature_dim);
    push_constants.eps = eps;
    

    VkCommandBuffer command_buffer = begin_single_time_commands();
    
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(command_buffer, &begin_info);
    
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_layer_norm_pipeline->get_pipeline());
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
                           m_layer_norm_pipeline->get_layout(), 0, 1, &descriptor_set, 0, nullptr);
    
    vkCmdPushConstants(command_buffer, m_layer_norm_pipeline->get_layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
    

    vkCmdDispatch(command_buffer, static_cast<uint32_t>(batch_size), static_cast<uint32_t>(seq_length), 1);
    
    end_single_time_commands(command_buffer);
    

    vkDestroyDescriptorPool(m_device->get_device(), descriptor_pool, nullptr);
    
    return true;
}

bool TensorOps::attention(Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value, float scale) {

    auto q_shape = query.shape();
    auto k_shape = key.shape();
    auto v_shape = value.shape();
    auto o_shape = output.shape();
    
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 || o_shape.size() != 4) {
        std::cerr << "Multi-head attention requires 4D tensors [batch, heads, seq_len, head_dim]" << std::endl;
        return false;
    }
    
    if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] || q_shape[0] != o_shape[0] ||
        q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1] || q_shape[1] != o_shape[1] ||
        q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2] || q_shape[2] != o_shape[2] ||
        q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] || q_shape[3] != o_shape[3]) {
        std::cerr << "Multi-head attention tensor shapes must match" << std::endl;
        return false;
    }
    
    if (!m_multi_head_attention_pipeline) {
        std::cerr << "Multi-head attention pipeline not available" << std::endl;
        return false;
    }
    
    uint32_t batch_size = q_shape[0];
    uint32_t num_heads = q_shape[1];
    uint32_t seq_length = q_shape[2];
    uint32_t head_dim = q_shape[3];
    uint32_t total_elements = batch_size * num_heads * seq_length * seq_length;
    

    VkDescriptorPool descriptor_pool;
    std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}  // 4 buffers: query, key, value, output
    };
    
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    
    if (vkCreateDescriptorPool(m_device->get_device(), &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool for multi-head attention" << std::endl;
        return false;
    }
    

    VkDescriptorSet descriptor_set;
    VkDescriptorSetLayout layout = m_multi_head_attention_pipeline->get_descriptor_layout();
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;
    
    if (vkAllocateDescriptorSets(m_device->get_device(), &alloc_info, &descriptor_set) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device->get_device(), descriptor_pool, nullptr);
        std::cerr << "Failed to allocate descriptor set for multi-head attention" << std::endl;
        return false;
    }
    

    std::vector<VkWriteDescriptorSet> descriptor_writes(4);
    
    VkDescriptorBufferInfo query_buffer_info{};
    query_buffer_info.buffer = query.buffer();
    query_buffer_info.offset = 0;
    query_buffer_info.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo key_buffer_info{};
    key_buffer_info.buffer = key.buffer();
    key_buffer_info.offset = 0;
    key_buffer_info.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo value_buffer_info{};
    value_buffer_info.buffer = value.buffer();
    value_buffer_info.offset = 0;
    value_buffer_info.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo output_buffer_info{};
    output_buffer_info.buffer = output.buffer();
    output_buffer_info.offset = 0;
    output_buffer_info.range = VK_WHOLE_SIZE;
    
    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = descriptor_set;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &query_buffer_info;
    
    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].dstSet = descriptor_set;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].pBufferInfo = &key_buffer_info;
    
    descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[2].dstSet = descriptor_set;
    descriptor_writes[2].dstBinding = 2;
    descriptor_writes[2].dstArrayElement = 0;
    descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[2].descriptorCount = 1;
    descriptor_writes[2].pBufferInfo = &value_buffer_info;
    
    descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[3].dstSet = descriptor_set;
    descriptor_writes[3].dstBinding = 3;
    descriptor_writes[3].dstArrayElement = 0;
    descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[3].descriptorCount = 1;
    descriptor_writes[3].pBufferInfo = &output_buffer_info;
    
    vkUpdateDescriptorSets(m_device->get_device(), 4, descriptor_writes.data(), 0, nullptr);
    

    VkCommandBuffer cmd = begin_single_time_commands();
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_multi_head_attention_pipeline->get_pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                           m_multi_head_attention_pipeline->get_layout(), 0, 1, &descriptor_set, 0, nullptr);
    

    struct PushConstants {
        uint32_t batch_size;
        uint32_t seq_length;
        uint32_t num_heads;
        uint32_t head_dim;
        uint32_t total_elements;
        float scale;
    } push_constants = {batch_size, seq_length, num_heads, head_dim, total_elements, scale};
    
    vkCmdPushConstants(cmd, m_multi_head_attention_pipeline->get_layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);
    

    uint32_t workgroup_size = 256;
    uint32_t num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
    vkCmdDispatch(cmd, num_workgroups, 1, 1);
    
    end_single_time_commands(cmd);
    

    vkDestroyDescriptorPool(m_device->get_device(), descriptor_pool, nullptr);
    
    return true;
}

bool TensorOps::reshape_for_attention(Tensor& output, const Tensor& input, 
                                     size_t batch_size, size_t seq_len, size_t num_heads, size_t head_dim) {
    if (!m_tensor_reshape_pipeline) {
        std::cerr << "Tensor reshape pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    

    m_tensor_reshape_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_tensor_reshape_pipeline->update_descriptor_set(0, 1, output.buffer());
    

    m_tensor_reshape_pipeline->bind(cmd);
    

    struct ReshapeConstants {
        uint32_t total_elements;
        uint32_t input_dim0, input_dim1, input_dim2, input_dim3;
        uint32_t output_dim0, output_dim1, output_dim2, output_dim3;
        uint32_t num_dims;
    } constants;
    
    constants.total_elements = static_cast<uint32_t>(input.size());
    constants.input_dim0 = static_cast<uint32_t>(batch_size);
    constants.input_dim1 = static_cast<uint32_t>(seq_len);
    constants.input_dim2 = static_cast<uint32_t>(num_heads * head_dim);  // embed_dim
    constants.input_dim3 = 1;
    constants.output_dim0 = static_cast<uint32_t>(batch_size);
    constants.output_dim1 = static_cast<uint32_t>(num_heads);
    constants.output_dim2 = static_cast<uint32_t>(seq_len);
    constants.output_dim3 = static_cast<uint32_t>(head_dim);
    constants.num_dims = 3;  // 3D to 4D reshape
    
    m_tensor_reshape_pipeline->push_constants(cmd, &constants, sizeof(constants));
    

    uint32_t num_workgroups = (constants.total_elements + 255) / 256;
    m_tensor_reshape_pipeline->dispatch(cmd, num_workgroups, 1, 1);
    
    end_single_time_commands(cmd);
    
    return true;
}

bool TensorOps::reshape_from_attention(Tensor& output, const Tensor& input,
                                      size_t batch_size, size_t seq_len, size_t num_heads, size_t head_dim) {
    if (!m_tensor_reshape_pipeline) {
        std::cerr << "Tensor reshape pipeline not initialized" << std::endl;
        return false;
    }
    
    VkCommandBuffer cmd = begin_single_time_commands();
    

    m_tensor_reshape_pipeline->update_descriptor_set(0, 0, input.buffer());
    m_tensor_reshape_pipeline->update_descriptor_set(0, 1, output.buffer());
    

    m_tensor_reshape_pipeline->bind(cmd);
    

    struct ReshapeConstants {
        uint32_t total_elements;
        uint32_t input_dim0, input_dim1, input_dim2, input_dim3;
        uint32_t output_dim0, output_dim1, output_dim2, output_dim3;
        uint32_t num_dims;
    } constants;
    
    constants.total_elements = static_cast<uint32_t>(input.size());
    constants.input_dim0 = static_cast<uint32_t>(batch_size);
    constants.input_dim1 = static_cast<uint32_t>(num_heads);
    constants.input_dim2 = static_cast<uint32_t>(seq_len);
    constants.input_dim3 = static_cast<uint32_t>(head_dim);
    constants.output_dim0 = static_cast<uint32_t>(batch_size);
    constants.output_dim1 = static_cast<uint32_t>(seq_len);
    constants.output_dim2 = static_cast<uint32_t>(num_heads * head_dim);  // embed_dim
    constants.output_dim3 = 1;
    constants.num_dims = 4;  // 4D to 3D reshape
    
    m_tensor_reshape_pipeline->push_constants(cmd, &constants, sizeof(constants));
    

    uint32_t num_workgroups = (constants.total_elements + 255) / 256;
    m_tensor_reshape_pipeline->dispatch(cmd, num_workgroups, 1, 1);
    
    end_single_time_commands(cmd);
    
    return true;
}


TensorOps* TensorOps::s_instance = nullptr;

bool TensorOps::initialize(VulkanDevice* device) {
    if (s_instance) {
        std::cerr << "TensorOps already initialized" << std::endl;
        return false;
    }
    
    std::shared_ptr<VulkanDevice> shared_device(device, [](VulkanDevice*) {

    });
    
    s_instance = new TensorOps(shared_device);
    return s_instance->initialize();
}

void TensorOps::shutdown() {
    if (s_instance) {
        delete s_instance;
        s_instance = nullptr;
    }
}

} // namespace dlvk
