#pragma once

#include <vector>
#include <memory>
#include <vulkan/vulkan.h>
#include "dlvk/tensor/tensor.h"
#include "dlvk/core/vulkan_device.h"

namespace dlvk {
namespace deployment {

/**
 * @brief Ring topology for distributed GPU communication
 */
struct RingNode {
    int device_id;
    int prev_device;  // device to receive from
    int next_device;  // device to send to
    VkCommandBuffer transfer_cmd;
    VkSemaphore send_semaphore;
    VkSemaphore recv_semaphore;
    VkFence completion_fence;
};

/**
 * @brief Tensor chunk for ring all-reduce operations
 */
struct TensorChunk {
    size_t offset;        // Start offset in tensor
    size_t size;          // Size in bytes
    VkBuffer src_buffer;  // Source buffer
    VkBuffer dst_buffer;  // Destination buffer
    VkDeviceMemory src_memory;
    VkDeviceMemory dst_memory;
};

/**
 * @brief Real ring all-reduce implementation for multi-GPU gradient sync
 */
class RingAllReduce {
public:
    explicit RingAllReduce(const std::vector<std::shared_ptr<VulkanDevice>>& devices);
    ~RingAllReduce();

    /**
     * @brief Perform ring all-reduce on tensors
     * @param tensors Tensors to reduce across all devices
     */
    void perform_all_reduce(std::vector<std::shared_ptr<Tensor>>& tensors);

    /**
     * @brief Get performance statistics
     */
    struct Stats {
        double total_time_ms = 0.0;
        double transfer_time_ms = 0.0;
        double compute_time_ms = 0.0;
        size_t bytes_transferred = 0;
        double bandwidth_gbps = 0.0;
    };
    
    Stats get_stats() const { return stats_; }

private:
    std::vector<std::shared_ptr<VulkanDevice>> devices_;
    std::vector<RingNode> ring_topology_;
    VkCommandPool command_pool_;
    VkDescriptorPool descriptor_pool_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline add_pipeline_;
    Stats stats_;

    void create_ring_topology();
    void create_vulkan_resources();
    std::vector<TensorChunk> split_tensor_into_chunks(std::shared_ptr<Tensor> tensor, size_t num_devices);
    void perform_reduce_scatter(std::vector<std::shared_ptr<Tensor>>& tensors);
    void perform_all_gather(std::vector<std::shared_ptr<Tensor>>& tensors);
    void transfer_chunk_to_next_device(const TensorChunk& chunk, int device_rank);
    void add_chunks_gpu(const TensorChunk& dst_chunk, const TensorChunk& src_chunk);
    void synchronize_all_devices();
};

} // namespace deployment
} // namespace dlvk
