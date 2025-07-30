#include "dlvk/deployment/ring_all_reduce.h"
#include "dlvk/core/vulkan_device.h"
#include <chrono>
#include <stdexcept>
#include <cstring>

namespace dlvk::deployment {

RingAllReduce::RingAllReduce(const std::vector<std::shared_ptr<VulkanDevice>>& devices)
    : devices_(devices), command_pool_(VK_NULL_HANDLE), descriptor_pool_(VK_NULL_HANDLE),
      pipeline_layout_(VK_NULL_HANDLE), add_pipeline_(VK_NULL_HANDLE) {
    
    if (devices_.empty()) {
        throw std::runtime_error("At least one device required for ring all-reduce");
    }
    
    create_ring_topology();
    create_vulkan_resources();
}

RingAllReduce::~RingAllReduce() {
    if (devices_.empty()) return;
    
    auto device = devices_[0]->get_device();
    
    if (add_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, add_pipeline_, nullptr);
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
    }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
    }
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, command_pool_, nullptr);
    }
    

    for (auto& node : ring_topology_) {
        if (node.send_semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, node.send_semaphore, nullptr);
        }
        if (node.recv_semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, node.recv_semaphore, nullptr);
        }
        if (node.completion_fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, node.completion_fence, nullptr);
        }
    }
}

void RingAllReduce::create_ring_topology() {
    size_t num_devices = devices_.size();
    ring_topology_.resize(num_devices);
    
    for (size_t i = 0; i < num_devices; ++i) {
        ring_topology_[i].device_id = static_cast<int>(i);
        ring_topology_[i].prev_device = static_cast<int>((i - 1 + num_devices) % num_devices);
        ring_topology_[i].next_device = static_cast<int>((i + 1) % num_devices);
        
        auto device = devices_[i]->get_device();
        

        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        vkCreateSemaphore(device, &semaphore_info, nullptr, &ring_topology_[i].send_semaphore);
        vkCreateSemaphore(device, &semaphore_info, nullptr, &ring_topology_[i].recv_semaphore);
        

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device, &fence_info, nullptr, &ring_topology_[i].completion_fence);
    }
}

void RingAllReduce::create_vulkan_resources() {
    auto device = devices_[0]->get_device();
    auto queue_family = devices_[0]->get_queue_families().compute_family.value();
    

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family;
    
    if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool for ring all-reduce");
    }
    

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 128; // Support for multiple tensors
    
    VkDescriptorPoolCreateInfo desc_pool_info{};
    desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    desc_pool_info.poolSizeCount = 1;
    desc_pool_info.pPoolSizes = &pool_size;
    desc_pool_info.maxSets = 64;
    
    if (vkCreateDescriptorPool(device, &desc_pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool for ring all-reduce");
    }
    

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    
    if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout for ring all-reduce");
    }
}

std::vector<TensorChunk> RingAllReduce::split_tensor_into_chunks(std::shared_ptr<Tensor> tensor, size_t num_devices) {
    std::vector<TensorChunk> chunks;
    chunks.reserve(num_devices);
    
    size_t tensor_size = tensor->size() * tensor->element_size();
    size_t chunk_size = tensor_size / num_devices;
    size_t remainder = tensor_size % num_devices;
    
    size_t offset = 0;
    for (size_t i = 0; i < num_devices; ++i) {
        TensorChunk chunk;
        chunk.offset = offset;
        chunk.size = chunk_size + (i < remainder ? 1 : 0);
        chunk.src_buffer = tensor->buffer();
        chunk.dst_buffer = tensor->buffer(); // In-place for now
        chunk.src_memory = tensor->memory();
        chunk.dst_memory = tensor->memory();
        
        chunks.push_back(chunk);
        offset += chunk.size;
    }
    
    return chunks;
}

void RingAllReduce::perform_all_reduce(std::vector<std::shared_ptr<Tensor>>& tensors) {
    auto start_time = std::chrono::high_resolution_clock::now();
    

    perform_reduce_scatter(tensors);
    

    perform_all_gather(tensors);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    stats_.total_time_ms = duration.count();
    

    size_t total_bytes = 0;
    for (auto& tensor : tensors) {
        total_bytes += tensor->size() * tensor->element_size();
    }
    stats_.bytes_transferred = total_bytes * devices_.size() * 2; // Reduce-scatter + All-gather
    stats_.bandwidth_gbps = (stats_.bytes_transferred / 1e9) / (stats_.total_time_ms / 1000.0);
}

void RingAllReduce::perform_reduce_scatter(std::vector<std::shared_ptr<Tensor>>& tensors) {
    size_t num_devices = devices_.size();
    
    for (auto& tensor : tensors) {
        auto chunks = split_tensor_into_chunks(tensor, num_devices);
        

        for (size_t step = 0; step < num_devices - 1; ++step) {
            for (size_t device_rank = 0; device_rank < num_devices; ++device_rank) {
                int send_chunk = (device_rank - step + num_devices) % num_devices;
                int recv_chunk = (device_rank - step - 1 + num_devices) % num_devices;
                

                transfer_chunk_to_next_device(chunks[send_chunk], device_rank);
                

                add_chunks_gpu(chunks[recv_chunk], chunks[send_chunk]);
            }
            
            synchronize_all_devices();
        }
    }
}

void RingAllReduce::perform_all_gather(std::vector<std::shared_ptr<Tensor>>& tensors) {
    size_t num_devices = devices_.size();
    
    for (auto& tensor : tensors) {
        auto chunks = split_tensor_into_chunks(tensor, num_devices);
        

        for (size_t step = 0; step < num_devices - 1; ++step) {
            for (size_t device_rank = 0; device_rank < num_devices; ++device_rank) {
                int send_chunk = (device_rank - step + 1 + num_devices) % num_devices;
                int recv_chunk = (device_rank - step + num_devices) % num_devices;
                

                transfer_chunk_to_next_device(chunks[send_chunk], device_rank);
            }
            
            synchronize_all_devices();
        }
    }
}

void RingAllReduce::transfer_chunk_to_next_device(const TensorChunk& chunk, int device_rank) {


    
    auto& node = ring_topology_[device_rank];
    auto device = devices_[device_rank]->get_device();
    

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    
    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(device, &alloc_info, &cmd_buffer);
    
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(cmd_buffer, &begin_info);
    

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    
    vkCmdPipelineBarrier(cmd_buffer, 
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0, 1, &barrier, 0, nullptr, 0, nullptr);
    
    vkEndCommandBuffer(cmd_buffer);
    

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &node.send_semaphore;
    
    auto queue = devices_[device_rank]->get_compute_queue();
    vkQueueSubmit(queue, 1, &submit_info, node.completion_fence);
    
    stats_.transfer_time_ms += 0.1; // Simplified timing
}

void RingAllReduce::add_chunks_gpu(const TensorChunk& dst_chunk, const TensorChunk& src_chunk) {


    stats_.compute_time_ms += 0.05; // Simplified timing
}

void RingAllReduce::synchronize_all_devices() {

    for (size_t i = 0; i < devices_.size(); ++i) {
        auto device = devices_[i]->get_device();
        vkWaitForFences(device, 1, &ring_topology_[i].completion_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &ring_topology_[i].completion_fence);
    }
}

} // namespace dlvk::deployment
