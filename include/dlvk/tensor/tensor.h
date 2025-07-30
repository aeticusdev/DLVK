#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
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
    


    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;


    const std::vector<size_t>& shape() const { return m_shape; }
    size_t size() const { return m_size; }
    size_t element_size() const;
    DataType dtype() const { return m_dtype; }
    std::shared_ptr<VulkanDevice> device() const { return m_device; }
    

    VkBuffer buffer() const { return m_buffer; }
    VkDeviceMemory memory() const { return m_memory; }
    

    void upload_data(const void* data);
    void download_data(void* data) const;
    

    std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape) const;
    std::shared_ptr<Tensor> transpose(const std::vector<size_t>& axes = {}) const;
    

    std::shared_ptr<Tensor> add(const Tensor& other) const;
    std::shared_ptr<Tensor> add_broadcast(const Tensor& other) const; // For bias addition
    std::shared_ptr<Tensor> subtract(const Tensor& other) const;
    std::shared_ptr<Tensor> multiply(const Tensor& other) const;
    std::shared_ptr<Tensor> multiply_scalar(float scalar) const; // Scalar multiplication
    std::shared_ptr<Tensor> divide(const Tensor& other) const;
    std::shared_ptr<Tensor> matrix_multiply(const Tensor& other) const;
    

    std::shared_ptr<Tensor> sum(int axis = -1) const;
    std::shared_ptr<Tensor> mean(int axis = -1) const;
    std::shared_ptr<Tensor> max(int axis = -1) const;
    std::shared_ptr<Tensor> min(int axis = -1) const;
    

    std::shared_ptr<Tensor> relu() const;
    std::shared_ptr<Tensor> sigmoid() const;
    std::shared_ptr<Tensor> tanh() const;
    std::shared_ptr<Tensor> softmax() const;
    

    std::shared_ptr<Tensor> relu_backward(const Tensor& grad_output) const; // 'this' is input
    std::shared_ptr<Tensor> sigmoid_backward(const Tensor& grad_output) const; // 'this' is output
    std::shared_ptr<Tensor> tanh_backward(const Tensor& grad_output) const; // 'this' is output


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
