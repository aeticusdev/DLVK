#pragma once

#include <memory>
#include <vulkan/vulkan.h>

namespace dlvk {

class VulkanDevice;
class Tensor;
class ComputePipeline;

class TensorOps {
public:
    TensorOps(std::shared_ptr<VulkanDevice> device);
    ~TensorOps();

    bool initialize();
    
    // Element-wise operations
    bool add(const Tensor& a, const Tensor& b, Tensor& result);
    bool add_broadcast(const Tensor& a, const Tensor& b, Tensor& result); // For broadcasting bias addition
    bool multiply(const Tensor& a, const Tensor& b, Tensor& result);
    bool subtract(const Tensor& a, const Tensor& b, Tensor& result);
    bool divide(const Tensor& a, const Tensor& b, Tensor& result);
    
    // Matrix operations
    bool matrix_multiply(const Tensor& a, const Tensor& b, Tensor& result);
    bool transpose(const Tensor& input, Tensor& result);
    
    // Activation functions
    bool relu(const Tensor& input, Tensor& result);
    bool sigmoid(const Tensor& input, Tensor& result);
    bool tanh_activation(const Tensor& input, Tensor& result);
    bool softmax(const Tensor& input, Tensor& result);
    
    // Reduction operations
    bool sum(const Tensor& input, Tensor& result, int axis = -1);
    bool mean(const Tensor& input, Tensor& result, int axis = -1);
    bool max(const Tensor& input, Tensor& result, int axis = -1);
    bool min(const Tensor& input, Tensor& result, int axis = -1);
    
    // Utility operations
    bool fill(Tensor& tensor, float value);
    bool copy(const Tensor& source, Tensor& destination);

private:
    std::shared_ptr<VulkanDevice> m_device;
    
    // Compute pipelines for different operations
    std::unique_ptr<ComputePipeline> m_add_pipeline;
    std::unique_ptr<ComputePipeline> m_multiply_pipeline;
    std::unique_ptr<ComputePipeline> m_subtract_pipeline;
    std::unique_ptr<ComputePipeline> m_divide_pipeline;
    std::unique_ptr<ComputePipeline> m_matmul_pipeline;
    std::unique_ptr<ComputePipeline> m_relu_pipeline;
    std::unique_ptr<ComputePipeline> m_sigmoid_pipeline;
    std::unique_ptr<ComputePipeline> m_tanh_pipeline;
    std::unique_ptr<ComputePipeline> m_softmax_pipeline;
    std::unique_ptr<ComputePipeline> m_transpose_pipeline;
    std::unique_ptr<ComputePipeline> m_reduce_sum_pipeline;
    std::unique_ptr<ComputePipeline> m_fill_pipeline;
    
    // Command buffer for operations
    VkCommandBuffer m_command_buffer = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    
    // Helper methods
    bool create_pipelines();
    bool allocate_command_buffer();
    VkCommandBuffer begin_single_time_commands();
    void end_single_time_commands(VkCommandBuffer cmd_buffer);
    
    // Validation helpers
    bool validate_element_wise_operation(const Tensor& a, const Tensor& b, const Tensor& result);
    bool validate_matrix_multiply(const Tensor& a, const Tensor& b, const Tensor& result);
};

} // namespace dlvk
