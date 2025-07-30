#include "dlvk/layers/recurrent_layers.h"
#include "dlvk/tensor/tensor_ops.h"
#include <random>
#include <cmath>

namespace dlvk {





RNNLayer::RNNLayer(size_t input_size, size_t hidden_size, std::shared_ptr<VulkanDevice> device)
    : m_input_size(input_size), m_hidden_size(hidden_size), m_device(device) {
    initialize_parameters();
    create_compute_pipelines();
    reset_hidden_state();
}

std::shared_ptr<Tensor> RNNLayer::forward(const std::shared_ptr<Tensor>& input) {

    auto input_shape = input->shape();
    size_t batch_size = input_shape[0];
    size_t sequence_length = input_shape[1];
    m_sequence_length = sequence_length;
    

    if (!m_hidden_state || m_hidden_state->shape()[0] != batch_size) {
        reset_hidden_state();
        auto hidden_shape = std::vector<size_t>{batch_size, m_hidden_size};
        m_hidden_state = std::make_shared<Tensor>(hidden_shape, DataType::FLOAT32, m_device);

        std::vector<float> zeros(batch_size * m_hidden_size, 0.0f);
        m_hidden_state->upload_data(zeros.data());
    }
    

    auto output_shape = std::vector<size_t>{batch_size, sequence_length, m_hidden_size};
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, m_device);
    



    TensorOps tensor_ops(m_device);
    tensor_ops.initialize();
    tensor_ops.fill(*output, 0.0f);
    
    return output;
}

std::shared_ptr<Tensor> RNNLayer::backward(const std::shared_ptr<Tensor>& grad_output) {


    auto input_shape = std::vector<size_t>{grad_output->shape()[0], grad_output->shape()[1], m_input_size};
    return std::make_shared<Tensor>(input_shape, DataType::FLOAT32, m_device);
}

void RNNLayer::reset_hidden_state() {
    m_hidden_state = nullptr;
}

void RNNLayer::set_hidden_state(std::shared_ptr<Tensor> hidden_state) {
    m_hidden_state = hidden_state;
}

std::unique_ptr<Layer> RNNLayer::clone() const {
    return std::make_unique<RNNLayer>(m_input_size, m_hidden_size, m_device);
}

void RNNLayer::initialize_parameters() {

    std::random_device rd;
    std::mt19937 gen(rd());
    

    float bound_ih = std::sqrt(6.0f / (m_input_size + m_hidden_size));
    float bound_hh = std::sqrt(6.0f / (m_hidden_size + m_hidden_size));
    
    std::uniform_real_distribution<float> dist_ih(-bound_ih, bound_ih);
    std::uniform_real_distribution<float> dist_hh(-bound_hh, bound_hh);
    

    auto weight_ih_shape = std::vector<size_t>{m_hidden_size, m_input_size};
    m_weight_ih = std::make_shared<Tensor>(weight_ih_shape, DataType::FLOAT32, m_device);
    std::vector<float> weight_ih_data(m_hidden_size * m_input_size);
    for (auto& w : weight_ih_data) w = dist_ih(gen);
    m_weight_ih->upload_data(weight_ih_data.data());
    

    auto weight_hh_shape = std::vector<size_t>{m_hidden_size, m_hidden_size};
    m_weight_hh = std::make_shared<Tensor>(weight_hh_shape, DataType::FLOAT32, m_device);
    std::vector<float> weight_hh_data(m_hidden_size * m_hidden_size);
    for (auto& w : weight_hh_data) w = dist_hh(gen);
    m_weight_hh->upload_data(weight_hh_data.data());
    

    auto bias_shape = std::vector<size_t>{m_hidden_size};
    m_bias_ih = std::make_shared<Tensor>(bias_shape, DataType::FLOAT32, m_device);
    m_bias_hh = std::make_shared<Tensor>(bias_shape, DataType::FLOAT32, m_device);
    
    std::vector<float> zeros(m_hidden_size, 0.0f);
    m_bias_ih->upload_data(zeros.data());
    m_bias_hh->upload_data(zeros.data());
}

void RNNLayer::create_compute_pipelines() {


}





LSTMLayer::LSTMLayer(size_t input_size, size_t hidden_size, std::shared_ptr<VulkanDevice> device)
    : m_input_size(input_size), m_hidden_size(hidden_size), m_device(device) {
    initialize_parameters();
    create_compute_pipelines();
    reset_states();
}

std::shared_ptr<Tensor> LSTMLayer::forward(const std::shared_ptr<Tensor>& input) {

    auto input_shape = input->shape();
    size_t batch_size = input_shape[0];
    size_t sequence_length = input_shape[1];
    m_sequence_length = sequence_length;
    

    if (!m_hidden_state || m_hidden_state->shape()[0] != batch_size) {
        auto state_shape = std::vector<size_t>{batch_size, m_hidden_size};
        m_hidden_state = std::make_shared<Tensor>(state_shape, DataType::FLOAT32, m_device);
        m_cell_state = std::make_shared<Tensor>(state_shape, DataType::FLOAT32, m_device);
        
        std::vector<float> zeros(batch_size * m_hidden_size, 0.0f);
        m_hidden_state->upload_data(zeros.data());
        m_cell_state->upload_data(zeros.data());
    }
    

    auto output_shape = std::vector<size_t>{batch_size, sequence_length, m_hidden_size};
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, m_device);
    


    TensorOps tensor_ops(m_device);
    tensor_ops.initialize();
    tensor_ops.fill(*output, 0.0f);
    
    return output;
}

std::shared_ptr<Tensor> LSTMLayer::backward(const std::shared_ptr<Tensor>& grad_output) {

    auto input_shape = std::vector<size_t>{grad_output->shape()[0], grad_output->shape()[1], m_input_size};
    return std::make_shared<Tensor>(input_shape, DataType::FLOAT32, m_device);
}

void LSTMLayer::reset_states() {
    m_hidden_state = nullptr;
    m_cell_state = nullptr;
}

void LSTMLayer::set_states(std::shared_ptr<Tensor> hidden_state, std::shared_ptr<Tensor> cell_state) {
    m_hidden_state = hidden_state;
    m_cell_state = cell_state;
}

std::unique_ptr<Layer> LSTMLayer::clone() const {
    return std::make_unique<LSTMLayer>(m_input_size, m_hidden_size, m_device);
}

void LSTMLayer::initialize_parameters() {
    std::random_device rd;
    std::mt19937 gen(rd());
    

    size_t gate_size = 4 * m_hidden_size;
    
    float bound_ih = std::sqrt(6.0f / (m_input_size + gate_size));
    float bound_hh = std::sqrt(6.0f / (m_hidden_size + gate_size));
    
    std::uniform_real_distribution<float> dist_ih(-bound_ih, bound_ih);
    std::uniform_real_distribution<float> dist_hh(-bound_hh, bound_hh);
    

    auto weight_ih_shape = std::vector<size_t>{gate_size, m_input_size};
    auto weight_hh_shape = std::vector<size_t>{gate_size, m_hidden_size};
    auto bias_shape = std::vector<size_t>{gate_size};
    
    m_weight_ih = std::make_shared<Tensor>(weight_ih_shape, DataType::FLOAT32, m_device);
    m_weight_hh = std::make_shared<Tensor>(weight_hh_shape, DataType::FLOAT32, m_device);
    m_bias_ih = std::make_shared<Tensor>(bias_shape, DataType::FLOAT32, m_device);
    m_bias_hh = std::make_shared<Tensor>(bias_shape, DataType::FLOAT32, m_device);
    

    std::vector<float> weight_ih_data(gate_size * m_input_size);
    std::vector<float> weight_hh_data(gate_size * m_hidden_size);
    
    for (auto& w : weight_ih_data) w = dist_ih(gen);
    for (auto& w : weight_hh_data) w = dist_hh(gen);
    
    m_weight_ih->upload_data(weight_ih_data.data());
    m_weight_hh->upload_data(weight_hh_data.data());
    

    std::vector<float> bias_data(gate_size, 0.0f);

    for (size_t i = m_hidden_size; i < 2 * m_hidden_size; ++i) {
        bias_data[i] = 1.0f;
    }
    
    m_bias_ih->upload_data(bias_data.data());
    m_bias_hh->upload_data(std::vector<float>(gate_size, 0.0f).data());
}

void LSTMLayer::create_compute_pipelines() {

}





GRULayer::GRULayer(size_t input_size, size_t hidden_size, std::shared_ptr<VulkanDevice> device)
    : m_input_size(input_size), m_hidden_size(hidden_size), m_device(device) {
    initialize_parameters();
    create_compute_pipelines();
    reset_hidden_state();
}

std::shared_ptr<Tensor> GRULayer::forward(const std::shared_ptr<Tensor>& input) {

    auto input_shape = input->shape();
    size_t batch_size = input_shape[0];
    size_t sequence_length = input_shape[1];
    m_sequence_length = sequence_length;
    

    if (!m_hidden_state || m_hidden_state->shape()[0] != batch_size) {
        auto state_shape = std::vector<size_t>{batch_size, m_hidden_size};
        m_hidden_state = std::make_shared<Tensor>(state_shape, DataType::FLOAT32, m_device);
        
        std::vector<float> zeros(batch_size * m_hidden_size, 0.0f);
        m_hidden_state->upload_data(zeros.data());
    }
    

    auto output_shape = std::vector<size_t>{batch_size, sequence_length, m_hidden_size};
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, m_device);
    


    TensorOps tensor_ops(m_device);
    tensor_ops.initialize();
    tensor_ops.fill(*output, 0.0f);
    
    return output;
}

std::shared_ptr<Tensor> GRULayer::backward(const std::shared_ptr<Tensor>& grad_output) {

    auto input_shape = std::vector<size_t>{grad_output->shape()[0], grad_output->shape()[1], m_input_size};
    return std::make_shared<Tensor>(input_shape, DataType::FLOAT32, m_device);
}

void GRULayer::reset_hidden_state() {
    m_hidden_state = nullptr;
}

void GRULayer::set_hidden_state(std::shared_ptr<Tensor> hidden_state) {
    m_hidden_state = hidden_state;
}

std::unique_ptr<Layer> GRULayer::clone() const {
    return std::make_unique<GRULayer>(m_input_size, m_hidden_size, m_device);
}

void GRULayer::initialize_parameters() {
    std::random_device rd;
    std::mt19937 gen(rd());
    

    size_t gate_size = 3 * m_hidden_size;
    
    float bound_ih = std::sqrt(6.0f / (m_input_size + gate_size));
    float bound_hh = std::sqrt(6.0f / (m_hidden_size + gate_size));
    
    std::uniform_real_distribution<float> dist_ih(-bound_ih, bound_ih);
    std::uniform_real_distribution<float> dist_hh(-bound_hh, bound_hh);
    

    auto weight_ih_shape = std::vector<size_t>{gate_size, m_input_size};
    auto weight_hh_shape = std::vector<size_t>{gate_size, m_hidden_size};
    auto bias_shape = std::vector<size_t>{gate_size};
    
    m_weight_ih = std::make_shared<Tensor>(weight_ih_shape, DataType::FLOAT32, m_device);
    m_weight_hh = std::make_shared<Tensor>(weight_hh_shape, DataType::FLOAT32, m_device);
    m_bias_ih = std::make_shared<Tensor>(bias_shape, DataType::FLOAT32, m_device);
    m_bias_hh = std::make_shared<Tensor>(bias_shape, DataType::FLOAT32, m_device);
    

    std::vector<float> weight_ih_data(gate_size * m_input_size);
    std::vector<float> weight_hh_data(gate_size * m_hidden_size);
    std::vector<float> bias_data(gate_size, 0.0f);
    
    for (auto& w : weight_ih_data) w = dist_ih(gen);
    for (auto& w : weight_hh_data) w = dist_hh(gen);
    
    m_weight_ih->upload_data(weight_ih_data.data());
    m_weight_hh->upload_data(weight_hh_data.data());
    m_bias_ih->upload_data(bias_data.data());
    m_bias_hh->upload_data(bias_data.data());
}

void GRULayer::create_compute_pipelines() {

}

} // namespace dlvk
