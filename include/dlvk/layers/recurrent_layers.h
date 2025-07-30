#pragma once

#include "layer.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/core/vulkan_device.h"
#include "dlvk/compute/compute_pipeline.h"
#include <memory>
#include <vector>

namespace dlvk {

/**
 * @brief Vanilla RNN layer implementation
 * 
 * Implements basic recurrent neural network with tanh activation:
 * h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
 */
class RNNLayer : public Layer {
public:
    RNNLayer(size_t input_size, size_t hidden_size, std::shared_ptr<VulkanDevice> device);
    ~RNNLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    void set_training(bool training) { m_training = training; }
    std::unique_ptr<Layer> clone() const override;
    

    void reset_hidden_state();
    std::shared_ptr<Tensor> get_hidden_state() const { return m_hidden_state; }
    void set_hidden_state(std::shared_ptr<Tensor> hidden_state);

private:
    size_t m_input_size;
    size_t m_hidden_size;
    size_t m_sequence_length;
    bool m_training = true;
    std::shared_ptr<VulkanDevice> m_device;
    

    std::shared_ptr<Tensor> m_weight_ih;  // Input to hidden weights
    std::shared_ptr<Tensor> m_weight_hh;  // Hidden to hidden weights  
    std::shared_ptr<Tensor> m_bias_ih;    // Input to hidden bias
    std::shared_ptr<Tensor> m_bias_hh;    // Hidden to hidden bias
    

    std::shared_ptr<Tensor> m_hidden_state;
    

    std::unique_ptr<ComputePipeline> m_rnn_forward_pipeline;
    std::unique_ptr<ComputePipeline> m_rnn_backward_pipeline;
    
    void initialize_parameters();
    void create_compute_pipelines();
};

/**
 * @brief LSTM (Long Short-Term Memory) layer implementation
 * 
 * Implements LSTM cell with forget, input, and output gates:
 * f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)  # Forget gate
 * i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)  # Input gate  
 * C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)    # Candidate values
 * C_t = f_t * C_{t-1} + i_t * C̃_t           # Cell state
 * o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)  # Output gate
 * h_t = o_t * tanh(C_t)                      # Hidden state
 */
class LSTMLayer : public Layer {
public:
    LSTMLayer(size_t input_size, size_t hidden_size, std::shared_ptr<VulkanDevice> device);
    ~LSTMLayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    void set_training(bool training) { m_training = training; }
    std::unique_ptr<Layer> clone() const override;
    

    void reset_states();
    std::shared_ptr<Tensor> get_hidden_state() const { return m_hidden_state; }
    std::shared_ptr<Tensor> get_cell_state() const { return m_cell_state; }
    void set_states(std::shared_ptr<Tensor> hidden_state, std::shared_ptr<Tensor> cell_state);

private:
    size_t m_input_size;
    size_t m_hidden_size;
    size_t m_sequence_length;
    bool m_training = true;
    std::shared_ptr<VulkanDevice> m_device;
    

    std::shared_ptr<Tensor> m_weight_ih;  // Input-hidden weights [4*hidden_size, input_size]
    std::shared_ptr<Tensor> m_weight_hh;  // Hidden-hidden weights [4*hidden_size, hidden_size]
    std::shared_ptr<Tensor> m_bias_ih;    // Input-hidden bias [4*hidden_size]
    std::shared_ptr<Tensor> m_bias_hh;    // Hidden-hidden bias [4*hidden_size]
    

    std::shared_ptr<Tensor> m_hidden_state;  // h_t
    std::shared_ptr<Tensor> m_cell_state;    // C_t
    

    std::shared_ptr<Tensor> m_input_gate;    // i_t
    std::shared_ptr<Tensor> m_forget_gate;   // f_t
    std::shared_ptr<Tensor> m_cell_gate;     // C̃_t
    std::shared_ptr<Tensor> m_output_gate;   // o_t
    

    std::unique_ptr<ComputePipeline> m_lstm_forward_pipeline;
    std::unique_ptr<ComputePipeline> m_lstm_backward_pipeline;
    
    void initialize_parameters();
    void create_compute_pipelines();
};

/**
 * @brief GRU (Gated Recurrent Unit) layer implementation
 * 
 * Implements GRU cell with reset and update gates:
 * r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)  # Reset gate
 * z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)  # Update gate
 * h̃_t = tanh(W_ih * x_t + b_ih + r_t * (W_hh * h_{t-1} + b_hh))  # New gate
 * h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t                   # Hidden state
 */
class GRULayer : public Layer {
public:
    GRULayer(size_t input_size, size_t hidden_size, std::shared_ptr<VulkanDevice> device);
    ~GRULayer() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    void set_training(bool training) { m_training = training; }
    std::unique_ptr<Layer> clone() const override;
    

    void reset_hidden_state();
    std::shared_ptr<Tensor> get_hidden_state() const { return m_hidden_state; }
    void set_hidden_state(std::shared_ptr<Tensor> hidden_state);

private:
    size_t m_input_size;
    size_t m_hidden_size;
    size_t m_sequence_length;
    bool m_training = true;
    std::shared_ptr<VulkanDevice> m_device;
    

    std::shared_ptr<Tensor> m_weight_ih;  // Input-hidden weights [3*hidden_size, input_size]
    std::shared_ptr<Tensor> m_weight_hh;  // Hidden-hidden weights [3*hidden_size, hidden_size]
    std::shared_ptr<Tensor> m_bias_ih;    // Input-hidden bias [3*hidden_size]
    std::shared_ptr<Tensor> m_bias_hh;    // Hidden-hidden bias [3*hidden_size]
    

    std::shared_ptr<Tensor> m_hidden_state;  // h_t
    

    std::shared_ptr<Tensor> m_reset_gate;    // r_t
    std::shared_ptr<Tensor> m_update_gate;   // z_t
    std::shared_ptr<Tensor> m_new_gate;      // h̃_t
    

    std::unique_ptr<ComputePipeline> m_gru_forward_pipeline;
    std::unique_ptr<ComputePipeline> m_gru_backward_pipeline;
    
    void initialize_parameters();
    void create_compute_pipelines();
};

/**
 * @brief Bidirectional RNN wrapper
 * 
 * Wraps any RNN layer to process sequences in both directions
 */
template<typename RNNType>
class BidirectionalRNN : public Layer {
public:
    BidirectionalRNN(size_t input_size, size_t hidden_size, std::shared_ptr<VulkanDevice> device);
    ~BidirectionalRNN() override = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    
    void set_training(bool training) override;
    std::unique_ptr<Layer> clone() const override;

private:
    std::unique_ptr<RNNType> m_forward_rnn;
    std::unique_ptr<RNNType> m_backward_rnn;
    size_t m_input_size;
    size_t m_hidden_size;
    
    std::shared_ptr<Tensor> reverse_sequence(std::shared_ptr<Tensor> input);
};


using BiLSTM = BidirectionalRNN<LSTMLayer>;
using BiGRU = BidirectionalRNN<GRULayer>;
using BiRNN = BidirectionalRNN<RNNLayer>;

} // namespace dlvk
