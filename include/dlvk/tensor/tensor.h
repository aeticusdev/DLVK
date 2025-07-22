#pragma once

#include <vector>
#include <memory>
#include <vulkan/vulkan.h>

namespace dlvk {

class VulkanDevice;
class TensorOps;

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT8
};

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, DataType dtype, std::shared_ptr<VulkanDevice> device);
    ~Tensor();

    // Shape and properties
    const std::vector<size_t>& shape() const { return m_shape; }
    size_t size() const { return m_size; }
    size_t element_size() const;
    DataType dtype() const { return m_dtype; }
    std::shared_ptr<VulkanDevice> device() const { return m_device; }
    
    // Memory management
    VkBuffer buffer() const { return m_buffer; }
    VkDeviceMemory memory() const { return m_memory; }
    
    // Data operations
    void upload_data(const void* data);
    void download_data(void* data) const;
    
    // Tensor operations (using static TensorOps instance)
    std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape) const;
    std::shared_ptr<Tensor> transpose(const std::vector<size_t>& axes = {}) const;
    
    // Arithmetic operations
    std::shared_ptr<Tensor> add(const Tensor& other) const;
    std::shared_ptr<Tensor> add_broadcast(const Tensor& other) const; // For bias addition
    std::shared_ptr<Tensor> subtract(const Tensor& other) const;
    std::shared_ptr<Tensor> multiply(const Tensor& other) const;
    std::shared_ptr<Tensor> divide(const Tensor& other) const;
    std::shared_ptr<Tensor> matrix_multiply(const Tensor& other) const;
    
    // Reduction operations
    std::shared_ptr<Tensor> sum(int axis = -1) const;
    std::shared_ptr<Tensor> mean(int axis = -1) const;
    std::shared_ptr<Tensor> max(int axis = -1) const;
    std::shared_ptr<Tensor> min(int axis = -1) const;
    
    // Activation functions
    std::shared_ptr<Tensor> relu() const;
    std::shared_ptr<Tensor> sigmoid() const;
    std::shared_ptr<Tensor> tanh() const;
    std::shared_ptr<Tensor> softmax() const;

    // Static tensor operations instance (shared across all tensors)
    static void set_tensor_ops(std::shared_ptr<TensorOps> ops);

private:
    std::vector<size_t> m_shape;
    size_t m_size;
    DataType m_dtype;
    std::shared_ptr<VulkanDevice> m_device;
    
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    
    static std::shared_ptr<TensorOps> s_tensor_ops;
    
    void allocate_memory();
    void deallocate_memory();
    
    size_t calculate_size() const;
};

} // namespace dlvk
