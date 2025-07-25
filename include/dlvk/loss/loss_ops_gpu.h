#pragma once

#include <memory>
#include <vulkan/vulkan.h>
#include "dlvk/tensor/tensor.h"

namespace dlvk {

class VulkanDevice;
class ComputePipeline;

class LossOpsGPU {
public:
    LossOpsGPU(std::shared_ptr<VulkanDevice> device);
    ~LossOpsGPU();

    bool initialize();
    
    // MSE Loss Operations
    bool mse_forward(const std::shared_ptr<Tensor>& predictions, 
                     const std::shared_ptr<Tensor>& targets,
                     std::shared_ptr<Tensor>& result);
    
    bool mse_backward(const std::shared_ptr<Tensor>& predictions,
                      const std::shared_ptr<Tensor>& targets,
                      std::shared_ptr<Tensor>& gradient);
    
    // Cross Entropy Loss Operations
    bool cross_entropy_forward(const std::shared_ptr<Tensor>& predictions,
                               const std::shared_ptr<Tensor>& targets,
                               std::shared_ptr<Tensor>& result);
    
    bool cross_entropy_backward(const std::shared_ptr<Tensor>& predictions,
                                const std::shared_ptr<Tensor>& targets,
                                std::shared_ptr<Tensor>& gradient);
    
    // Binary Cross Entropy Loss Operations
    bool binary_cross_entropy_forward(const std::shared_ptr<Tensor>& predictions,
                                       const std::shared_ptr<Tensor>& targets,
                                       std::shared_ptr<Tensor>& result,
                                       float epsilon = 1e-7f);
    
    bool binary_cross_entropy_backward(const std::shared_ptr<Tensor>& predictions,
                                        const std::shared_ptr<Tensor>& targets,
                                        std::shared_ptr<Tensor>& gradient,
                                        float epsilon = 1e-7f);

private:
    std::shared_ptr<VulkanDevice> m_device;
    
    // Command buffer management
    VkCommandBuffer m_command_buffer = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    
    // GPU Pipelines
    std::unique_ptr<ComputePipeline> m_mse_forward_pipeline;
    std::unique_ptr<ComputePipeline> m_mse_backward_pipeline;
    std::unique_ptr<ComputePipeline> m_cross_entropy_forward_pipeline;
    std::unique_ptr<ComputePipeline> m_cross_entropy_backward_pipeline;
    std::unique_ptr<ComputePipeline> m_binary_cross_entropy_forward_pipeline;
    std::unique_ptr<ComputePipeline> m_binary_cross_entropy_backward_pipeline;
    
    bool allocate_command_buffer();
    bool create_pipelines();
    VkCommandBuffer begin_single_time_commands();
    void end_single_time_commands(VkCommandBuffer cmd);
};

} // namespace dlvk
