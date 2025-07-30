#pragma once

#include "layer.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/core/vulkan_device.h"
#include "dlvk/compute/compute_pipeline.h"
#include <memory>
#include <vector>

namespace dlvk {

/**
 * @brief Embedding layer for converting discrete tokens to dense vectors
 * 
 * Essential for NLP tasks, converts integer indices to learnable dense embeddings.
 */
class EmbeddingLayer : public Layer {
public:
    EmbeddingLayer(size_t num_embeddings, size_t embedding_dim, std::shared_ptr<VulkanDevice> device);
    ~EmbeddingLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    std::unique_ptr<Layer> clone() const override;
    
private:
    size_t m_num_embeddings;  // Vocabulary size
    size_t m_embedding_dim;   // Embedding dimension
    std::shared_ptr<Tensor> m_embeddings;  // Embedding table [num_embeddings, embedding_dim]
    
    void initialize_parameters();
    std::unique_ptr<ComputePipeline> m_embedding_pipeline;
};

/**
 * @brief Layer Normalization - essential for modern deep networks
 * 
 * Normalizes across the feature dimension, critical for Transformers and stable training.
 */
class LayerNormLayer : public Layer {
public:
    LayerNormLayer(size_t normalized_shape, float eps = 1e-5f, std::shared_ptr<VulkanDevice> device = nullptr);
    ~LayerNormLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    std::unique_ptr<Layer> clone() const override;
    
private:
    size_t m_normalized_shape;
    float m_eps;
    std::shared_ptr<Tensor> m_weight;  // Learnable scale parameter
    std::shared_ptr<Tensor> m_bias;    // Learnable shift parameter
    std::shared_ptr<Tensor> m_last_input;  // Stored for backward pass
    
    void initialize_parameters();
    std::unique_ptr<ComputePipeline> m_layer_norm_pipeline;
};

/**
 * @brief Multi-Head Attention mechanism
 * 
 * Core component of Transformers, enables the model to attend to different positions.
 */
class MultiHeadAttentionLayer : public Layer {
public:
    MultiHeadAttentionLayer(size_t embed_dim, size_t num_heads, 
                           float dropout = 0.0f, std::shared_ptr<VulkanDevice> device = nullptr);
    ~MultiHeadAttentionLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& query,
                                  const std::shared_ptr<Tensor>& key,
                                  const std::shared_ptr<Tensor>& value,
                                  const std::shared_ptr<Tensor>& attn_mask = nullptr);
    
    std::unique_ptr<Layer> clone() const override;
    
private:
    size_t m_embed_dim;
    size_t m_num_heads;
    size_t m_head_dim;
    float m_dropout;
    

    std::shared_ptr<Tensor> m_q_proj_weight;
    std::shared_ptr<Tensor> m_k_proj_weight;
    std::shared_ptr<Tensor> m_v_proj_weight;
    std::shared_ptr<Tensor> m_out_proj_weight;
    

    void reshape_for_attention(Tensor& output, const Tensor& input, 
                             size_t batch_size, size_t seq_len, 
                             size_t num_heads, size_t head_dim);
    void reshape_from_attention(Tensor& output, const Tensor& input,
                               size_t batch_size, size_t seq_len,
                               size_t num_heads, size_t head_dim);
    
    std::shared_ptr<Tensor> m_q_proj_bias;
    std::shared_ptr<Tensor> m_k_proj_bias;
    std::shared_ptr<Tensor> m_v_proj_bias;
    std::shared_ptr<Tensor> m_out_proj_bias;
    

    std::shared_ptr<Tensor> m_w_q;
    std::shared_ptr<Tensor> m_w_k;
    std::shared_ptr<Tensor> m_w_v;
    std::shared_ptr<Tensor> m_w_o;
    
    void initialize_parameters();
    std::unique_ptr<ComputePipeline> m_attention_pipeline;
    
    std::shared_ptr<Tensor> scaled_dot_product_attention(
        const std::shared_ptr<Tensor>& query,
        const std::shared_ptr<Tensor>& key,
        const std::shared_ptr<Tensor>& value,
        const std::shared_ptr<Tensor>& attn_mask = nullptr
    );
};

/**
 * @brief Transformer Encoder Layer
 * 
 * Complete transformer block with multi-head attention and feed-forward network.
 */
class TransformerEncoderLayer : public Layer {
public:
    TransformerEncoderLayer(size_t d_model, size_t nhead, size_t dim_feedforward = 2048,
                           float dropout = 0.1f, std::shared_ptr<VulkanDevice> device = nullptr);
    ~TransformerEncoderLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    std::unique_ptr<Layer> clone() const override;
    
private:
    size_t m_d_model;
    size_t m_nhead;
    size_t m_dim_feedforward;
    float m_dropout;
    
    std::unique_ptr<MultiHeadAttentionLayer> m_self_attn;
    std::unique_ptr<LayerNormLayer> m_norm1;
    std::unique_ptr<LayerNormLayer> m_norm2;
    

    std::shared_ptr<Tensor> m_linear1_weight;
    std::shared_ptr<Tensor> m_linear1_bias;
    std::shared_ptr<Tensor> m_linear2_weight;
    std::shared_ptr<Tensor> m_linear2_bias;
    
    void initialize_parameters();
};

/**
 * @brief Positional Encoding for Transformers
 * 
 * Adds positional information to input embeddings using sinusoidal encoding.
 */
class PositionalEncodingLayer : public Layer {
public:
    PositionalEncodingLayer(size_t d_model, size_t max_len = 5000, 
                           std::shared_ptr<VulkanDevice> device = nullptr);
    ~PositionalEncodingLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    std::unique_ptr<Layer> clone() const override;
    
private:
    size_t m_d_model;
    size_t m_max_len;
    std::shared_ptr<Tensor> m_pe;  // Precomputed positional encodings
    
    void initialize_encoding();
};

/**
 * @brief Group Normalization layer
 * 
 * Alternative to batch normalization that works well with small batch sizes.
 */
class GroupNormLayer : public Layer {
public:
    GroupNormLayer(size_t num_groups, size_t num_channels, float eps = 1e-5f,
                   std::shared_ptr<VulkanDevice> device = nullptr);
    ~GroupNormLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    std::unique_ptr<Layer> clone() const override;
    
private:
    size_t m_num_groups;
    size_t m_num_channels;
    float m_eps;
    std::shared_ptr<Tensor> m_weight;
    std::shared_ptr<Tensor> m_bias;
    
    void initialize_parameters();
    std::unique_ptr<ComputePipeline> m_group_norm_pipeline;
};

/**
 * @brief Residual/Skip Connection wrapper
 * 
 * Implements residual connections that are fundamental to deep networks.
 */
class ResidualLayer : public Layer {
public:
    ResidualLayer(std::unique_ptr<Layer> wrapped_layer);
    ~ResidualLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    std::unique_ptr<Layer> clone() const override;
    
private:
    std::unique_ptr<Layer> m_wrapped_layer;
    std::shared_ptr<Tensor> m_residual_input;  // Store input for residual connection
    std::unique_ptr<Layer> m_inner_layer;  // Alternative name used in implementation
};

/**
 * @brief Global Average Pooling layer
 * 
 * Pools spatial dimensions to single values, commonly used before classification.
 */
class GlobalAvgPoolLayer : public Layer {
public:
    GlobalAvgPoolLayer(std::shared_ptr<VulkanDevice> device = nullptr);
    ~GlobalAvgPoolLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    std::unique_ptr<Layer> clone() const override;
    
private:
    std::shared_ptr<Tensor> m_last_input;  // Store input for backward pass
    std::unique_ptr<ComputePipeline> m_global_pool_pipeline;
};

} // namespace dlvk
