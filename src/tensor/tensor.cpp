#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/core/vulkan_device.h"
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace dlvk {


std::shared_ptr<TensorOps> Tensor::s_tensor_ops = nullptr;

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype, std::shared_ptr<VulkanDevice> device)
    : m_shape(shape), m_dtype(dtype), m_device(device) {
    m_size = calculate_size();
    allocate_memory();
}

Tensor::~Tensor() {
    deallocate_memory();
}


Tensor::Tensor(const Tensor& other)
    : m_shape(other.m_shape)
    , m_dtype(other.m_dtype) 
    , m_device(other.m_device)
    , m_size(other.m_size)
    , m_buffer(VK_NULL_HANDLE)
    , m_memory(VK_NULL_HANDLE) {

    allocate_memory();
    

    if (other.m_buffer != VK_NULL_HANDLE && m_buffer != VK_NULL_HANDLE) {

        void* src_data;
        void* dst_data;
        VkDevice device = m_device->get_device();
        

        VkResult src_result = vkMapMemory(device, other.m_memory, 0, VK_WHOLE_SIZE, 0, &src_data);
        if (src_result == VK_SUCCESS) {

            VkResult dst_result = vkMapMemory(device, m_memory, 0, VK_WHOLE_SIZE, 0, &dst_data);
            if (dst_result == VK_SUCCESS) {

                std::memcpy(dst_data, src_data, m_size * element_size());
                vkUnmapMemory(device, m_memory);
            }
            vkUnmapMemory(device, other.m_memory);
        }
    }
}


Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {

        deallocate_memory();
        

        m_shape = other.m_shape;
        m_dtype = other.m_dtype;
        m_device = other.m_device;
        m_size = other.m_size;
        m_buffer = VK_NULL_HANDLE;
        m_memory = VK_NULL_HANDLE;
        

        allocate_memory();
        if (other.m_buffer != VK_NULL_HANDLE && m_buffer != VK_NULL_HANDLE) {
            void* src_data;
            void* dst_data;
            VkDevice device = m_device->get_device();
            
            VkResult src_result = vkMapMemory(device, other.m_memory, 0, VK_WHOLE_SIZE, 0, &src_data);
            if (src_result == VK_SUCCESS) {
                VkResult dst_result = vkMapMemory(device, m_memory, 0, VK_WHOLE_SIZE, 0, &dst_data);
                if (dst_result == VK_SUCCESS) {
                    std::memcpy(dst_data, src_data, m_size * element_size());
                    vkUnmapMemory(device, m_memory);
                }
                vkUnmapMemory(device, other.m_memory);
            }
        }
    }
    return *this;
}


Tensor::Tensor(Tensor&& other) noexcept
    : m_shape(std::move(other.m_shape))
    , m_dtype(other.m_dtype)
    , m_device(std::move(other.m_device))
    , m_size(other.m_size)
    , m_buffer(other.m_buffer)
    , m_memory(other.m_memory) {

    other.m_buffer = VK_NULL_HANDLE;
    other.m_memory = VK_NULL_HANDLE;
    other.m_size = 0;
}


Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {

        deallocate_memory();
        

        m_shape = std::move(other.m_shape);
        m_dtype = other.m_dtype;
        m_device = std::move(other.m_device);
        m_size = other.m_size;
        m_buffer = other.m_buffer;
        m_memory = other.m_memory;
        

        other.m_buffer = VK_NULL_HANDLE;
        other.m_memory = VK_NULL_HANDLE;
        other.m_size = 0;
    }
    return *this;
}

size_t Tensor::element_size() const {
    switch (m_dtype) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::FLOAT16: return sizeof(uint16_t);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT8: return sizeof(int8_t);
        default: throw std::runtime_error("Unknown data type");
    }
}

void Tensor::upload_data(const void* data) {
    void* mapped_memory;
    vkMapMemory(m_device->get_device(), m_memory, 0, m_size * element_size(), 0, &mapped_memory);
    std::memcpy(mapped_memory, data, m_size * element_size());
    vkUnmapMemory(m_device->get_device(), m_memory);
}

void Tensor::download_data(void* data) const {
    void* mapped_memory;
    vkMapMemory(m_device->get_device(), m_memory, 0, m_size * element_size(), 0, &mapped_memory);
    std::memcpy(data, mapped_memory, m_size * element_size());
    vkUnmapMemory(m_device->get_device(), m_memory);
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    if (new_size != m_size) {
        throw std::runtime_error("New shape must have the same number of elements");
    }
    
    auto result = std::make_shared<Tensor>(new_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        s_tensor_ops->copy(*this, *result);
    } else {

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = m_device->get_command_pool();
        alloc_info.commandBufferCount = 1;
        
        vkAllocateCommandBuffers(m_device->get_device(), &alloc_info, &cmd);
        
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkBeginCommandBuffer(cmd, &begin_info);
        
        VkBufferCopy copy_region{};
        copy_region.size = m_size * element_size();
        vkCmdCopyBuffer(cmd, m_buffer, result->m_buffer, 1, &copy_region);
        
        vkEndCommandBuffer(cmd);
        
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        
        vkQueueSubmit(m_device->get_compute_queue(), 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_device->get_compute_queue());
        
        vkFreeCommandBuffers(m_device->get_device(), m_device->get_command_pool(), 1, &cmd);
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::transpose(const std::vector<size_t>& axes) const {

    if (m_shape.size() != 2) {
        throw std::runtime_error("Transpose currently only supports 2D tensors");
    }
    
    std::vector<size_t> new_shape = {m_shape[1], m_shape[0]};
    auto result = std::make_shared<Tensor>(new_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->transpose(*this, *result)) {
            throw std::runtime_error("Failed to perform tensor transpose");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::add(const Tensor& other) const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->add(*this, other, *result)) {
            throw std::runtime_error("Failed to perform tensor addition");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::add_broadcast(const Tensor& other) const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->add_broadcast(*this, other, *result)) {
            throw std::runtime_error("Failed to perform broadcast tensor addition");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::multiply(const Tensor& other) const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->multiply(*this, other, *result)) {
            throw std::runtime_error("Failed to perform tensor multiplication");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::matrix_multiply(const Tensor& other) const {
    if (m_shape.size() != 2 || other.m_shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication requires 2D tensors");
    }
    
    if (m_shape[1] != other.m_shape[0]) {
        throw std::runtime_error("Invalid dimensions for matrix multiplication");
    }
    
    std::vector<size_t> result_shape = {m_shape[0], other.m_shape[1]};
    auto result = std::make_shared<Tensor>(result_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->matrix_multiply(*this, other, *result)) {
            throw std::runtime_error("Failed to perform matrix multiplication");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::relu() const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->relu(*this, *result)) {
            throw std::runtime_error("Failed to perform ReLU activation");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->sigmoid(*this, *result)) {
            throw std::runtime_error("Failed to perform Sigmoid activation");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::tanh() const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->tanh_activation(*this, *result)) {
            throw std::runtime_error("Failed to perform Tanh activation");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

void Tensor::set_tensor_ops(std::shared_ptr<TensorOps> ops) {
    s_tensor_ops = ops;
}

void Tensor::allocate_memory() {
    VkDeviceSize buffer_size = m_size * element_size();
    
    VkResult result = m_device->create_buffer(
        buffer_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_buffer,
        m_memory
    );
    
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create tensor buffer");
    }
}

void Tensor::deallocate_memory() {
    if (m_buffer != VK_NULL_HANDLE && m_memory != VK_NULL_HANDLE) {
        m_device->destroy_buffer(m_buffer, m_memory);
        m_buffer = VK_NULL_HANDLE;
        m_memory = VK_NULL_HANDLE;
    }
}

size_t Tensor::calculate_size() const {
    return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<size_t>());
}

std::shared_ptr<Tensor> Tensor::subtract(const Tensor& other) const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->subtract(*this, other, *result)) {
            throw std::runtime_error("Failed to perform tensor subtraction");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::multiply_scalar(float scalar) const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    

    if (s_tensor_ops && s_tensor_ops->scalar_multiply(*this, scalar, *result)) {
        return result;
    }
    

    std::vector<float> data(m_size);
    std::vector<float> result_data(m_size);
    
    download_data(data.data());
    
    for (size_t i = 0; i < m_size; ++i) {
        result_data[i] = data[i] * scalar;
    }
    
    result->upload_data(result_data.data());
    
    return result;
}

std::shared_ptr<Tensor> Tensor::divide(const Tensor& other) const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->divide(*this, other, *result)) {
            throw std::runtime_error("Failed to perform tensor division");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::sum(int axis) const {
    std::vector<size_t> result_shape = {1};  // For now, just total sum
    auto result = std::make_shared<Tensor>(result_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->sum(*this, *result, axis)) {
            throw std::runtime_error("Failed to perform tensor sum");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::mean(int axis) const {
    std::vector<size_t> result_shape = {1};  // For now, just total mean
    auto result = std::make_shared<Tensor>(result_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->mean(*this, *result, axis)) {
            throw std::runtime_error("Failed to perform tensor mean");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::max(int axis) const {
    std::vector<size_t> result_shape = {1};  // For now, just total max
    auto result = std::make_shared<Tensor>(result_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->max(*this, *result, axis)) {
            throw std::runtime_error("Failed to perform tensor max");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::min(int axis) const {
    std::vector<size_t> result_shape = {1};  // For now, just total min
    auto result = std::make_shared<Tensor>(result_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->min(*this, *result, axis)) {
            throw std::runtime_error("Failed to perform tensor min");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::softmax() const {
    auto result = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->softmax(*this, *result)) {
            throw std::runtime_error("Failed to perform tensor softmax");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return result;
}


std::shared_ptr<Tensor> Tensor::relu_backward(const Tensor& grad_output) const {
    auto grad_input = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->relu_backward(*this, grad_output, *grad_input)) {
            throw std::runtime_error("Failed to perform ReLU backward pass");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return grad_input;
}

std::shared_ptr<Tensor> Tensor::sigmoid_backward(const Tensor& grad_output) const {
    auto grad_input = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->sigmoid_backward(*this, grad_output, *grad_input)) {
            throw std::runtime_error("Failed to perform Sigmoid backward pass");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return grad_input;
}

std::shared_ptr<Tensor> Tensor::tanh_backward(const Tensor& grad_output) const {
    auto grad_input = std::make_shared<Tensor>(m_shape, m_dtype, m_device);
    
    if (s_tensor_ops) {
        if (!s_tensor_ops->tanh_backward(*this, grad_output, *grad_input)) {
            throw std::runtime_error("Failed to perform Tanh backward pass");
        }
    } else {
        throw std::runtime_error("TensorOps not initialized. Call Tensor::set_tensor_ops() first.");
    }
    
    return grad_input;
}

} // namespace dlvk
