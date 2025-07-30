#pragma once

#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <cmath>

namespace dlvk {



class UltraFastAttention {
private:
    VkDevice m_device;
    VkQueue m_queue;
    VkCommandBuffer m_cmd;
    VkPipeline m_pipeline;
    VkPipelineLayout m_layout;
    VkDescriptorSet m_desc_set;
    
public:

    inline UltraFastAttention(VkDevice device, VkQueue queue, VkCommandBuffer cmd, 
                             VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet desc_set)
        : m_device(device), m_queue(queue), m_cmd(cmd), m_pipeline(pipeline), m_layout(layout), m_desc_set(desc_set) {}
    

    inline bool execute_fused_attention(
        VkBuffer output_buffer, VkBuffer query_buffer, VkBuffer key_buffer, VkBuffer value_buffer,
        uint32_t batch_size, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim, float scale) {
        

        vkResetCommandBuffer(m_cmd, 0);
        

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(m_cmd, &begin_info);
        

        vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout, 0, 1, &m_desc_set, 0, nullptr);
        

        VkDescriptorBufferInfo buffer_infos[4] = {
            {query_buffer, 0, VK_WHOLE_SIZE},
            {key_buffer, 0, VK_WHOLE_SIZE}, 
            {value_buffer, 0, VK_WHOLE_SIZE},
            {output_buffer, 0, VK_WHOLE_SIZE}
        };
        
        VkWriteDescriptorSet writes[4];
        for (int i = 0; i < 4; i++) {
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_desc_set, (uint32_t)i, 0, 1, 
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[i], nullptr};
        }
        vkUpdateDescriptorSets(m_device, 4, writes, 0, nullptr);
        

        struct PushConstants {
            uint32_t batch_size, seq_len, num_heads, head_dim;
            float scale;
        } push_constants = {batch_size, seq_len, num_heads, head_dim, scale};
        
        vkCmdPushConstants(m_cmd, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
        

        uint32_t total_elements = batch_size * seq_len * num_heads * head_dim;
        uint32_t workgroups = (total_elements + 1023) / 1024;  // 1024 threads per workgroup for max occupancy
        vkCmdDispatch(m_cmd, workgroups, 1, 1);
        

        vkEndCommandBuffer(m_cmd);
        
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &m_cmd;
        

        VkResult result = vkQueueSubmit(m_queue, 1, &submit_info, VK_NULL_HANDLE);
        

        vkQueueWaitIdle(m_queue);
        
        return (result == VK_SUCCESS);
    }
    

    inline void fast_matmul(VkBuffer a, VkBuffer b, VkBuffer c, uint32_t m, uint32_t n, uint32_t k) {

    }
    

    inline void fast_softmax(VkBuffer input, VkBuffer output, uint32_t size) {

    }
};


class UltraFastMultiHeadAttention {
private:
    UltraFastAttention* m_attention;
    VkBuffer m_q_buffer, m_k_buffer, m_v_buffer, m_output_buffer;
    uint32_t m_batch_size, m_seq_len, m_num_heads, m_head_dim;
    float m_scale;
    
public:

    inline UltraFastMultiHeadAttention(UltraFastAttention* attention, 
                                      VkBuffer q_buf, VkBuffer k_buf, VkBuffer v_buf, VkBuffer out_buf,
                                      uint32_t batch, uint32_t seq, uint32_t heads, uint32_t head_dim)
        : m_attention(attention), m_q_buffer(q_buf), m_k_buffer(k_buf), m_v_buffer(v_buf), m_output_buffer(out_buf),
          m_batch_size(batch), m_seq_len(seq), m_num_heads(heads), m_head_dim(head_dim),
          m_scale(1.0f / sqrtf((float)head_dim)) {}
    

    inline bool forward() {

        return m_attention->execute_fused_attention(
            m_output_buffer, m_q_buffer, m_k_buffer, m_v_buffer,
            m_batch_size, m_seq_len, m_num_heads, m_head_dim, m_scale
        );
    }
};

} // namespace dlvk
