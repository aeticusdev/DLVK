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
    
    // Backward pass for activation functions
    bool relu_backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
    bool sigmoid_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input);
    bool tanh_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input);
    
    // Reduction operations
    bool sum(const Tensor& input, Tensor& result, int axis = -1);
    bool sum_axis0(const Tensor& input, Tensor& result); // Sum along batch dimension
    bool mean(const Tensor& input, Tensor& result, int axis = -1);
    bool max(const Tensor& input, Tensor& result, int axis = -1);
    bool min(const Tensor& input, Tensor& result, int axis = -1);
    
    // Utility operations
    bool fill(Tensor& tensor, float value);
    bool copy(const Tensor& source, Tensor& destination);
    
    // Scalar operations for optimizers
    bool scale(const Tensor& input, float scalar, Tensor& result);
    bool scalar_add(const Tensor& input, float scalar, Tensor& result);
    bool element_wise_multiply(const Tensor& a, const Tensor& b, Tensor& result);
    bool element_wise_sqrt(const Tensor& input, Tensor& result);
    bool element_wise_square(const Tensor& input, Tensor& result);
    bool adam_update(const Tensor& gradient, const Tensor& m, const Tensor& v, 
                     Tensor& param, Tensor& new_m, Tensor& new_v,
                     float lr, float beta1, float beta2, float epsilon);
    
    // GPU-based gradient clipping operations
    bool gradient_clip_by_norm(const Tensor& gradient, float max_norm, Tensor& clipped_gradient);
    bool gradient_clip_by_value(const Tensor& gradient, float min_val, float max_val, Tensor& clipped_gradient);
    
    // Static interface for global instance
    static bool initialize(VulkanDevice* device);
    static void shutdown();
    static TensorOps* instance() { return s_instance; }
    
    // CNN operations
    bool conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output,
                size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w);
    bool conv2d_backward_input(const Tensor& grad_output, const Tensor& weights, Tensor& grad_input,
                               size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w);
    bool conv2d_backward_weight(const Tensor& input, const Tensor& grad_output, 
                                Tensor& grad_weights, Tensor& grad_bias,
                                size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w);
    
    // Pooling operations
    bool maxpool2d(const Tensor& input, Tensor& output, Tensor& indices,
                   size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                   size_t padding_h, size_t padding_w);
    bool maxpool2d_backward(const Tensor& grad_output, const Tensor& indices, Tensor& grad_input);
    bool avgpool2d(const Tensor& input, Tensor& output,
                   size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                   size_t padding_h, size_t padding_w);
    bool avgpool2d_backward(const Tensor& grad_output, Tensor& grad_input,
                            size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                            size_t padding_h, size_t padding_w);
    
    // Batch normalization operations
    bool batch_norm(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                    Tensor& running_mean, Tensor& running_var,
                    Tensor& output, Tensor& saved_mean, Tensor& saved_var,
                    float momentum, float epsilon, bool training);
    bool batch_norm_backward(const Tensor& grad_output, const Tensor& input,
                             const Tensor& gamma, const Tensor& saved_mean, const Tensor& saved_var,
                             Tensor& grad_input, Tensor& grad_gamma, Tensor& grad_beta,
                             float epsilon);
    
    // Dropout operations
    bool dropout(const Tensor& input, Tensor& output, Tensor& mask,
                 float dropout_rate, bool training, uint32_t seed);
    bool dropout_backward(const Tensor& grad_output, const Tensor& mask, Tensor& grad_input,
                          float dropout_rate);

private:
    std::shared_ptr<VulkanDevice> m_device;
    
    // Static instance for global access
    static TensorOps* s_instance;
    
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
    
    // Backward pass pipelines
    std::unique_ptr<ComputePipeline> m_relu_backward_pipeline;
    std::unique_ptr<ComputePipeline> m_sigmoid_backward_pipeline;
    std::unique_ptr<ComputePipeline> m_tanh_backward_pipeline;
    
    // Specialized reduction pipelines
    std::unique_ptr<ComputePipeline> m_reduce_sum_axis0_pipeline;
    
    // CNN pipelines
    std::unique_ptr<ComputePipeline> m_conv2d_pipeline;
    std::unique_ptr<ComputePipeline> m_conv2d_backward_input_pipeline;
    std::unique_ptr<ComputePipeline> m_conv2d_backward_weight_pipeline;
    
    // Pooling pipelines
    std::unique_ptr<ComputePipeline> m_maxpool2d_pipeline;
    std::unique_ptr<ComputePipeline> m_maxpool2d_backward_pipeline;
    std::unique_ptr<ComputePipeline> m_avgpool2d_pipeline;
    std::unique_ptr<ComputePipeline> m_avgpool2d_backward_pipeline;
    
    // Batch normalization pipelines
    std::unique_ptr<ComputePipeline> m_batch_norm_pipeline;
    std::unique_ptr<ComputePipeline> m_batch_norm_backward_pipeline;
    
    // Dropout pipelines
    std::unique_ptr<ComputePipeline> m_dropout_pipeline;
    std::unique_ptr<ComputePipeline> m_dropout_backward_pipeline;
    
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
