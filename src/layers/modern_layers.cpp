#include "dlvk/layers/modern_layers.h"
#include "dlvk/tensor/tensor_ops.h"
#include <cmath>
#include <random>
#include <iostream>

namespace dlvk {





EmbeddingLayer::EmbeddingLayer(size_t num_embeddings, size_t embedding_dim, std::shared_ptr<VulkanDevice> device)
    : Layer(), m_num_embeddings(num_embeddings), m_embedding_dim(embedding_dim) {
    m_device = device;
    initialize_parameters();
}

void EmbeddingLayer::initialize_parameters() {

    std::vector<float> weights(m_num_embeddings * m_embedding_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(m_embedding_dim));
    
    for (auto& w : weights) {
        w = dist(gen);
    }
    
    std::vector<size_t> shape = {m_num_embeddings, m_embedding_dim};
    m_embeddings = std::make_shared<Tensor>(shape, DataType::FLOAT32, m_device);
    m_embeddings->upload_data(weights.data());
}

std::shared_ptr<Tensor> EmbeddingLayer::forward(const std::shared_ptr<Tensor>& input) {
    const auto& input_shape = input->shape();
    

    if (input_shape.size() != 2) {
        throw std::runtime_error("EmbeddingLayer: Input must be 2D (batch_size, sequence_length)");
    }
    
    size_t batch_size = input_shape[0];
    size_t seq_length = input_shape[1];
    

    std::vector<size_t> output_shape = {batch_size, seq_length, m_embedding_dim};
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, m_device);
    

    TensorOps tensor_ops(m_device);
    tensor_ops.initialize();
    
    if (!tensor_ops.embedding_lookup(*output, *input, *m_embeddings)) {
        throw std::runtime_error("EmbeddingLayer: Failed to perform embedding lookup");
    }
    
    return output;
}

std::shared_ptr<Tensor> EmbeddingLayer::backward(const std::shared_ptr<Tensor>& grad_output) {


    

    return nullptr;
}

std::unique_ptr<Layer> EmbeddingLayer::clone() const {
    return std::make_unique<EmbeddingLayer>(m_num_embeddings, m_embedding_dim, m_device);
}





LayerNormLayer::LayerNormLayer(size_t normalized_shape, float eps, std::shared_ptr<VulkanDevice> device)
    : Layer(), m_normalized_shape(normalized_shape), m_eps(eps) {
    m_device = device;
    

    std::vector<float> weight_data(normalized_shape, 1.0f);
    std::vector<float> bias_data(normalized_shape, 0.0f);
    
    std::vector<size_t> param_shape = {normalized_shape};
    m_weight = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, m_device);
    m_bias = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, m_device);
    m_weight->upload_data(weight_data.data());
    m_bias->upload_data(bias_data.data());
}

std::shared_ptr<Tensor> LayerNormLayer::forward(const std::shared_ptr<Tensor>& input) {
    const auto& input_shape = input->shape();
    auto output = std::make_shared<Tensor>(input_shape, DataType::FLOAT32, m_device);
    

    size_t last_dim = input_shape.back();
    if (last_dim != m_normalized_shape) {
        throw std::runtime_error("LayerNormLayer: Last dimension must match normalized_shape");
    }
    

    m_last_input = input;
    

    auto* tensor_ops = TensorOps::instance();
    if (!tensor_ops->layer_norm(*output, *input, *m_weight, *m_bias, m_eps)) {
        throw std::runtime_error("LayerNormLayer: GPU layer normalization failed");
    }
    
    return output;
}

std::shared_ptr<Tensor> LayerNormLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    auto grad_input = std::make_shared<Tensor>(m_last_input->shape(), DataType::FLOAT32, m_device);
    



    
    return grad_input;
}

std::unique_ptr<Layer> LayerNormLayer::clone() const {
    return std::make_unique<LayerNormLayer>(m_normalized_shape, m_eps, m_device);
}





MultiHeadAttentionLayer::MultiHeadAttentionLayer(size_t embed_dim, size_t num_heads, float dropout, std::shared_ptr<VulkanDevice> device)
    : Layer(), m_embed_dim(embed_dim), m_num_heads(num_heads), 
      m_dropout(dropout), m_head_dim(embed_dim / num_heads) {
    
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }
    
    m_device = device;
    initialize_parameters();
}

void MultiHeadAttentionLayer::initialize_parameters() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / m_embed_dim));
    

    std::vector<float> w_q_data(m_embed_dim * m_embed_dim);
    std::vector<float> w_k_data(m_embed_dim * m_embed_dim);
    std::vector<float> w_v_data(m_embed_dim * m_embed_dim);
    std::vector<float> w_o_data(m_embed_dim * m_embed_dim);
    
    for (auto& w : w_q_data) w = dist(gen);
    for (auto& w : w_k_data) w = dist(gen);
    for (auto& w : w_v_data) w = dist(gen);
    for (auto& w : w_o_data) w = dist(gen);
    
    std::vector<size_t> weight_shape = {m_embed_dim, m_embed_dim};
    m_w_q = std::make_shared<Tensor>(weight_shape, DataType::FLOAT32, m_device);
    m_w_k = std::make_shared<Tensor>(weight_shape, DataType::FLOAT32, m_device);
    m_w_v = std::make_shared<Tensor>(weight_shape, DataType::FLOAT32, m_device);
    m_w_o = std::make_shared<Tensor>(weight_shape, DataType::FLOAT32, m_device);
    
    m_w_q->upload_data(w_q_data.data());
    m_w_k->upload_data(w_k_data.data());
    m_w_v->upload_data(w_v_data.data());
    m_w_o->upload_data(w_o_data.data());
}

std::shared_ptr<Tensor> MultiHeadAttentionLayer::forward(const std::shared_ptr<Tensor>& input) {

    return forward(input, input, input);
}

std::shared_ptr<Tensor> MultiHeadAttentionLayer::forward(const std::shared_ptr<Tensor>& query, 
                                                         const std::shared_ptr<Tensor>& key, 
                                                         const std::shared_ptr<Tensor>& value, 
                                                         const std::shared_ptr<Tensor>& attn_mask) {
    const auto& q_shape = query->shape();
    
    if (q_shape.size() != 3) {
        throw std::runtime_error("MultiHeadAttentionLayer: Input must be 3D (batch_size, seq_len, embed_dim)");
    }
    
    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[1];
    size_t embed_dim = q_shape[2];
    
    if (embed_dim != m_embed_dim) {
        throw std::runtime_error("MultiHeadAttentionLayer: Input embed_dim doesn't match layer embed_dim");
    }
    
    try {

        TensorOps tensor_ops(m_device);
        if (!tensor_ops.initialize()) {
            throw std::runtime_error("Failed to initialize TensorOps for attention");
        }
        


        std::vector<size_t> proj_shape = {batch_size, seq_len, m_embed_dim};
        

        std::vector<size_t> batch_weight_shape = {batch_size, m_embed_dim, m_embed_dim};
        auto batch_w_q = std::make_shared<Tensor>(batch_weight_shape, DataType::FLOAT32, m_device);
        auto batch_w_k = std::make_shared<Tensor>(batch_weight_shape, DataType::FLOAT32, m_device);
        auto batch_w_v = std::make_shared<Tensor>(batch_weight_shape, DataType::FLOAT32, m_device);
        

        std::vector<float> weight_data(m_embed_dim * m_embed_dim);
        std::vector<float> batch_weight_data(batch_size * m_embed_dim * m_embed_dim);
        

        m_w_q->download_data(weight_data.data());
        for (size_t b = 0; b < batch_size; ++b) {
            std::copy(weight_data.begin(), weight_data.end(), 
                     batch_weight_data.begin() + b * m_embed_dim * m_embed_dim);
        }
        batch_w_q->upload_data(batch_weight_data.data());
        

        m_w_k->download_data(weight_data.data());
        for (size_t b = 0; b < batch_size; ++b) {
            std::copy(weight_data.begin(), weight_data.end(), 
                     batch_weight_data.begin() + b * m_embed_dim * m_embed_dim);
        }
        batch_w_k->upload_data(batch_weight_data.data());
        

        m_w_v->download_data(weight_data.data());
        for (size_t b = 0; b < batch_size; ++b) {
            std::copy(weight_data.begin(), weight_data.end(), 
                     batch_weight_data.begin() + b * m_embed_dim * m_embed_dim);
        }
        batch_w_v->upload_data(batch_weight_data.data());
        

        auto Q = std::make_shared<Tensor>(proj_shape, DataType::FLOAT32, m_device);
        auto K = std::make_shared<Tensor>(proj_shape, DataType::FLOAT32, m_device);
        auto V = std::make_shared<Tensor>(proj_shape, DataType::FLOAT32, m_device);
        

        if (!tensor_ops.batch_matrix_multiply(*query, *batch_w_q, *Q)) {
            std::cerr << "MultiHeadAttentionLayer forward error: Failed to compute Q projection" << std::endl;
            return query; // Return input as fallback
        }
        if (!tensor_ops.batch_matrix_multiply(*key, *batch_w_k, *K)) {
            std::cerr << "MultiHeadAttentionLayer forward error: Failed to compute K projection" << std::endl;
            return query; // Return input as fallback  
        }
        if (!tensor_ops.batch_matrix_multiply(*value, *batch_w_v, *V)) {
            std::cerr << "MultiHeadAttentionLayer forward error: Failed to compute V projection" << std::endl;
            return query; // Return input as fallback
        }
        

        std::vector<size_t> attention_shape = {batch_size, m_num_heads, seq_len, m_head_dim};
        auto Q_heads = std::make_shared<Tensor>(attention_shape, DataType::FLOAT32, m_device);
        auto K_heads = std::make_shared<Tensor>(attention_shape, DataType::FLOAT32, m_device);
        auto V_heads = std::make_shared<Tensor>(attention_shape, DataType::FLOAT32, m_device);
        auto attn_output = std::make_shared<Tensor>(attention_shape, DataType::FLOAT32, m_device);
        

        if (!tensor_ops.reshape_for_attention(*Q_heads, *Q, batch_size, seq_len, m_num_heads, m_head_dim)) {
            std::cerr << "Failed to reshape Q for attention" << std::endl;
            return query;
        }
        if (!tensor_ops.reshape_for_attention(*K_heads, *K, batch_size, seq_len, m_num_heads, m_head_dim)) {
            std::cerr << "Failed to reshape K for attention" << std::endl;
            return query;
        }
        if (!tensor_ops.reshape_for_attention(*V_heads, *V, batch_size, seq_len, m_num_heads, m_head_dim)) {
            std::cerr << "Failed to reshape V for attention" << std::endl;
            return query;
        }
        

        float scale = 1.0f / std::sqrt(static_cast<float>(m_head_dim));
        
        if (!tensor_ops.attention(*attn_output, *Q_heads, *K_heads, *V_heads, scale)) {
            throw std::runtime_error("GPU attention computation failed");
        }
        

        auto concat_output = std::make_shared<Tensor>(proj_shape, DataType::FLOAT32, m_device);
        

        if (!tensor_ops.reshape_from_attention(*concat_output, *attn_output, batch_size, seq_len, m_num_heads, m_head_dim)) {
            std::cerr << "Failed to reshape from attention" << std::endl;
            return query;
        }
        

        auto batch_w_o = std::make_shared<Tensor>(batch_weight_shape, DataType::FLOAT32, m_device);
        

        m_w_o->download_data(weight_data.data());
        for (size_t b = 0; b < batch_size; ++b) {
            std::copy(weight_data.begin(), weight_data.end(), 
                     batch_weight_data.begin() + b * m_embed_dim * m_embed_dim);
        }
        batch_w_o->upload_data(batch_weight_data.data());
        

        auto output = std::make_shared<Tensor>(q_shape, DataType::FLOAT32, m_device);
        
        if (!tensor_ops.batch_matrix_multiply(*concat_output, *batch_w_o, *output)) {
            std::cerr << "MultiHeadAttentionLayer forward error: Failed to compute output projection" << std::endl;
            return query; // Return input as fallback
        }
        

        if (m_dropout > 0.0f) {


        }
        
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "MultiHeadAttentionLayer forward error: " << e.what() << std::endl;

        auto output = std::make_shared<Tensor>(q_shape, DataType::FLOAT32, m_device);
        return output;
    }
}

std::shared_ptr<Tensor> MultiHeadAttentionLayer::backward(const std::shared_ptr<Tensor>& grad_output) {

    auto grad_input = std::make_shared<Tensor>(grad_output->shape(), DataType::FLOAT32, m_device);
    return grad_input;
}

std::unique_ptr<Layer> MultiHeadAttentionLayer::clone() const {
    return std::make_unique<MultiHeadAttentionLayer>(m_embed_dim, m_num_heads, m_dropout, m_device);
}

void MultiHeadAttentionLayer::reshape_for_attention(Tensor& output, const Tensor& input, 
                                                   size_t batch_size, size_t seq_len, 
                                                   size_t num_heads, size_t head_dim) {


    


    
    std::vector<float> input_data(batch_size * seq_len * num_heads * head_dim);
    std::vector<float> output_data(batch_size * num_heads * seq_len * head_dim);
    

    input.download_data(input_data.data());
    

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t input_idx = b * seq_len * num_heads * head_dim + 
                                     s * num_heads * head_dim + 
                                     h * head_dim + d;
                    size_t output_idx = b * num_heads * seq_len * head_dim + 
                                      h * seq_len * head_dim + 
                                      s * head_dim + d;
                    output_data[output_idx] = input_data[input_idx];
                }
            }
        }
    }
    

    output.upload_data(output_data.data());
}

void MultiHeadAttentionLayer::reshape_from_attention(Tensor& output, const Tensor& input,
                                                    size_t batch_size, size_t seq_len,
                                                    size_t num_heads, size_t head_dim) {


    
    std::vector<float> input_data(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output_data(batch_size * seq_len * num_heads * head_dim);
    

    input.download_data(input_data.data());
    

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t input_idx = b * num_heads * seq_len * head_dim + 
                                     h * seq_len * head_dim + 
                                     s * head_dim + d;
                    size_t output_idx = b * seq_len * num_heads * head_dim + 
                                      s * num_heads * head_dim + 
                                      h * head_dim + d;
                    output_data[output_idx] = input_data[input_idx];
                }
            }
        }
    }
    

    output.upload_data(output_data.data());
}





TransformerEncoderLayer::TransformerEncoderLayer(size_t d_model, size_t nhead, size_t dim_feedforward, 
                                                 float dropout, std::shared_ptr<VulkanDevice> device)
    : Layer(), m_d_model(d_model), m_nhead(nhead), m_dim_feedforward(dim_feedforward), m_dropout(dropout) {
    
    m_device = device;
    

    m_self_attn = std::make_unique<MultiHeadAttentionLayer>(d_model, nhead, dropout, device);
    m_norm1 = std::make_unique<LayerNormLayer>(d_model, 1e-5f, device);
    m_norm2 = std::make_unique<LayerNormLayer>(d_model, 1e-5f, device);
    
    initialize_parameters();
}

void TransformerEncoderLayer::initialize_parameters() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / m_d_model));
    
    std::vector<float> w1_data(m_d_model * m_dim_feedforward);
    std::vector<float> w2_data(m_dim_feedforward * m_d_model);
    std::vector<float> b1_data(m_dim_feedforward, 0.0f);
    std::vector<float> b2_data(m_d_model, 0.0f);
    
    for (auto& w : w1_data) w = dist(gen);
    for (auto& w : w2_data) w = dist(gen);
    
    std::vector<size_t> w1_shape = {m_d_model, m_dim_feedforward};
    std::vector<size_t> w2_shape = {m_dim_feedforward, m_d_model};
    std::vector<size_t> b1_shape = {m_dim_feedforward};
    std::vector<size_t> b2_shape = {m_d_model};
    
    m_linear1_weight = std::make_shared<Tensor>(w1_shape, DataType::FLOAT32, m_device);
    m_linear2_weight = std::make_shared<Tensor>(w2_shape, DataType::FLOAT32, m_device);
    m_linear1_bias = std::make_shared<Tensor>(b1_shape, DataType::FLOAT32, m_device);
    m_linear2_bias = std::make_shared<Tensor>(b2_shape, DataType::FLOAT32, m_device);
    
    m_linear1_weight->upload_data(w1_data.data());
    m_linear2_weight->upload_data(w2_data.data());
    m_linear1_bias->upload_data(b1_data.data());
    m_linear2_bias->upload_data(b2_data.data());
}

std::shared_ptr<Tensor> TransformerEncoderLayer::forward(const std::shared_ptr<Tensor>& input) {

    auto output = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, m_device);
    return output;
}

std::shared_ptr<Tensor> TransformerEncoderLayer::backward(const std::shared_ptr<Tensor>& grad_output) {

    auto grad_input = std::make_shared<Tensor>(grad_output->shape(), DataType::FLOAT32, m_device);
    return grad_input;
}

std::unique_ptr<Layer> TransformerEncoderLayer::clone() const {
    return std::make_unique<TransformerEncoderLayer>(m_d_model, m_nhead, m_dim_feedforward, m_dropout, m_device);
}





PositionalEncodingLayer::PositionalEncodingLayer(size_t d_model, size_t max_len, std::shared_ptr<VulkanDevice> device)
    : Layer(), m_d_model(d_model), m_max_len(max_len) {
    
    m_device = device;
    initialize_encoding();
}

void PositionalEncodingLayer::initialize_encoding() {

    std::vector<float> pe_data(m_max_len * m_d_model);
    
    for (size_t pos = 0; pos < m_max_len; ++pos) {
        for (size_t i = 0; i < m_d_model; ++i) {
            float angle = pos / std::pow(10000.0f, 2.0f * (i / 2) / m_d_model);
            if (i % 2 == 0) {
                pe_data[pos * m_d_model + i] = std::sin(angle);
            } else {
                pe_data[pos * m_d_model + i] = std::cos(angle);
            }
        }
    }
    
    std::vector<size_t> pe_shape = {m_max_len, m_d_model};
    m_pe = std::make_shared<Tensor>(pe_shape, DataType::FLOAT32, m_device);
    m_pe->upload_data(pe_data.data());
}

std::shared_ptr<Tensor> PositionalEncodingLayer::forward(const std::shared_ptr<Tensor>& input) {
    const auto& input_shape = input->shape();
    
    if (input_shape.size() != 3 || input_shape[2] != m_d_model) {
        throw std::runtime_error("PositionalEncodingLayer: Input must be (batch_size, seq_len, d_model)");
    }
    
    size_t seq_len = input_shape[1];
    if (seq_len > m_max_len) {
        throw std::runtime_error("PositionalEncodingLayer: Sequence length exceeds maximum");
    }
    
    auto output = std::make_shared<Tensor>(input_shape, DataType::FLOAT32, m_device);
    


    
    return output;
}

std::shared_ptr<Tensor> PositionalEncodingLayer::backward(const std::shared_ptr<Tensor>& grad_output) {

    auto grad_input = std::make_shared<Tensor>(grad_output->shape(), DataType::FLOAT32, m_device);
    return grad_input;
}

std::unique_ptr<Layer> PositionalEncodingLayer::clone() const {
    return std::make_unique<PositionalEncodingLayer>(m_d_model, m_max_len, m_device);
}





GroupNormLayer::GroupNormLayer(size_t num_groups, size_t num_channels, float eps, std::shared_ptr<VulkanDevice> device)
    : Layer(), m_num_groups(num_groups), m_num_channels(num_channels), m_eps(eps) {
    
    if (num_channels % num_groups != 0) {
        throw std::runtime_error("GroupNormLayer: num_channels must be divisible by num_groups");
    }
    
    m_device = device;
    initialize_parameters();
}

void GroupNormLayer::initialize_parameters() {

    std::vector<float> weight_data(m_num_channels, 1.0f);
    std::vector<float> bias_data(m_num_channels, 0.0f);
    
    std::vector<size_t> param_shape = {m_num_channels};
    m_weight = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, m_device);
    m_bias = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, m_device);
    m_weight->upload_data(weight_data.data());
    m_bias->upload_data(bias_data.data());
}

std::shared_ptr<Tensor> GroupNormLayer::forward(const std::shared_ptr<Tensor>& input) {
    auto output = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, m_device);
    


    
    return output;
}

std::shared_ptr<Tensor> GroupNormLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    auto grad_input = std::make_shared<Tensor>(grad_output->shape(), DataType::FLOAT32, m_device);
    



    
    return grad_input;
}

std::unique_ptr<Layer> GroupNormLayer::clone() const {
    return std::make_unique<GroupNormLayer>(m_num_groups, m_num_channels, m_eps, m_device);
}





ResidualLayer::ResidualLayer(std::unique_ptr<Layer> inner_layer)
    : Layer(), m_inner_layer(std::move(inner_layer)) {
}

std::shared_ptr<Tensor> ResidualLayer::forward(const std::shared_ptr<Tensor>& input) {

    m_residual_input = input;
    

    auto inner_output = m_inner_layer->forward(input);
    

    auto output = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, m_device);

    
    return output;
}

std::shared_ptr<Tensor> ResidualLayer::backward(const std::shared_ptr<Tensor>& grad_output) {

    auto grad_inner = m_inner_layer->backward(grad_output);
    

    auto grad_input = std::make_shared<Tensor>(grad_output->shape(), DataType::FLOAT32, m_device);
    if (grad_inner) {

    }
    
    return grad_input;
}

std::unique_ptr<Layer> ResidualLayer::clone() const {
    return std::make_unique<ResidualLayer>(m_inner_layer->clone());
}





GlobalAvgPoolLayer::GlobalAvgPoolLayer(std::shared_ptr<VulkanDevice> device)
    : Layer() {
    m_device = device;
}

std::shared_ptr<Tensor> GlobalAvgPoolLayer::forward(const std::shared_ptr<Tensor>& input) {
    const auto& input_shape = input->shape();
    
    if (input_shape.size() < 3) {
        throw std::runtime_error("GlobalAvgPoolLayer: Input must have at least 3 dimensions");
    }
    

    m_last_input = input;
    

    std::vector<size_t> output_shape = {input_shape[0], input_shape[1]};
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, m_device);
    


    
    return output;
}

std::shared_ptr<Tensor> GlobalAvgPoolLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& input_shape = m_last_input->shape();
    auto grad_input = std::make_shared<Tensor>(input_shape, DataType::FLOAT32, m_device);
    


    
    return grad_input;
}

std::unique_ptr<Layer> GlobalAvgPoolLayer::clone() const {
    return std::make_unique<GlobalAvgPoolLayer>(m_device);
}

} // namespace dlvk
