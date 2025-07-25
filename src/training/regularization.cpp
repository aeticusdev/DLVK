#include "dlvk/training/regularization.h"
#include "dlvk/tensor/tensor_ops.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace dlvk {
namespace training {

// L1Regularizer implementation
float L1Regularizer::compute_loss(const std::vector<Tensor>& weights) {
    float total_loss = 0.0f;
    
    for (const auto& weight : weights) {
        // L1 norm calculation = sum(|w|)
        // This would ideally use GPU operations for absolute values and reduction
        std::vector<float> weight_data(weight.size());
        weight.download_data(weight_data.data());
        
        float weight_l1 = 0.0f;
        for (float val : weight_data) {
            weight_l1 += std::abs(val);
        }
        total_loss += weight_l1;
    }
    
    return m_lambda * total_loss;
}

std::vector<Tensor> L1Regularizer::compute_gradients(const std::vector<Tensor>& weights) {
    std::vector<Tensor> gradients;
    gradients.reserve(weights.size());
    
    for (const auto& weight : weights) {
        // L1 gradient = lambda * sign(w)
        Tensor grad(weight.shape(), weight.dtype(), weight.device());
        
        // Download weights, compute sign, upload back
        std::vector<float> weight_data(weight.size());
        weight.download_data(weight_data.data());
        
        std::vector<float> grad_data(weight.size());
        for (size_t i = 0; i < weight_data.size(); ++i) {
            grad_data[i] = m_lambda * (weight_data[i] > 0 ? 1.0f : (weight_data[i] < 0 ? -1.0f : 0.0f));
        }
        
        grad.upload_data(grad_data.data());
        gradients.push_back(grad);
    }
    
    return gradients;
}

// L2Regularizer implementation
float L2Regularizer::compute_loss(const std::vector<Tensor>& weights) {
    float total_loss = 0.0f;
    
    for (const auto& weight : weights) {
        // L2 norm calculation = sum(w^2)
        std::vector<float> weight_data(weight.size());
        weight.download_data(weight_data.data());
        
        float weight_l2 = 0.0f;
        for (float val : weight_data) {
            weight_l2 += val * val;
        }
        total_loss += weight_l2;
    }
    
    return 0.5f * m_lambda * total_loss;
}

std::vector<Tensor> L2Regularizer::compute_gradients(const std::vector<Tensor>& weights) {
    std::vector<Tensor> gradients;
    gradients.reserve(weights.size());
    
    for (const auto& weight : weights) {
        // L2 gradient = lambda * w
        Tensor grad(weight.shape(), weight.dtype(), weight.device());
        
        // Use TensorOps to scale weights by lambda
        auto* tensor_ops = TensorOps::instance();
        tensor_ops->scalar_multiply(weight, m_lambda, grad);
        gradients.push_back(grad);
    }
    
    return gradients;
}

// ElasticNetRegularizer implementation
float ElasticNetRegularizer::compute_loss(const std::vector<Tensor>& weights) {
    float l1_loss = m_l1_reg.compute_loss(weights);
    float l2_loss = m_l2_reg.compute_loss(weights);
    
    return l1_loss + l2_loss;
}

std::vector<Tensor> ElasticNetRegularizer::compute_gradients(const std::vector<Tensor>& weights) {
    auto l1_grads = m_l1_reg.compute_gradients(weights);
    auto l2_grads = m_l2_reg.compute_gradients(weights);
    
    std::vector<Tensor> combined_grads;
    combined_grads.reserve(weights.size());
    
    for (size_t i = 0; i < weights.size(); ++i) {
        // Add L1 and L2 gradients together
        Tensor combined(weights[i].shape(), weights[i].dtype(), weights[i].device());
        auto* tensor_ops = TensorOps::instance();
        tensor_ops->add(l1_grads[i], l2_grads[i], combined);
        combined_grads.push_back(combined);
    }
    
    return combined_grads;
}

void ElasticNetRegularizer::set_strength(float strength) {
    m_lambda = strength;
    m_l1_reg.set_strength(strength * m_l1_ratio);
    m_l2_reg.set_strength(strength * (1.0f - m_l1_ratio));
}

void ElasticNetRegularizer::set_l1_ratio(float ratio) {
    m_l1_ratio = std::clamp(ratio, 0.0f, 1.0f);
    m_l1_reg.set_strength(m_lambda * m_l1_ratio);
    m_l2_reg.set_strength(m_lambda * (1.0f - m_l1_ratio));
}

// WeightDecayRegularizer implementation
float WeightDecayRegularizer::compute_loss(const std::vector<Tensor>& weights) {
    // Weight decay doesn't contribute to loss directly
    return 0.0f;
}

std::vector<Tensor> WeightDecayRegularizer::compute_gradients(const std::vector<Tensor>& weights) {
    std::vector<Tensor> gradients;
    gradients.reserve(weights.size());
    
    for (const auto& weight : weights) {
        // Weight decay gradient = decay_rate * w
        Tensor grad(weight.shape(), weight.dtype(), weight.device());
        auto* tensor_ops = TensorOps::instance();
        tensor_ops->scalar_multiply(weight, m_decay_rate, grad);
        gradients.push_back(grad);
    }
    
    return gradients;
}

void WeightDecayRegularizer::apply_decay(std::vector<Tensor>& weights, float learning_rate) {
    for (auto& weight : weights) {
        // Apply weight decay directly: w = w * (1 - lr * decay_rate)
        float scale_factor = 1.0f - learning_rate * m_decay_rate;
        auto* tensor_ops = TensorOps::instance();
        tensor_ops->scalar_multiply(weight, scale_factor, weight);
    }
}

// AdvancedDropout implementation
void AdvancedDropout::update_rate(int epoch, float validation_loss) {
    m_current_epoch = epoch;
    
    if (epoch < m_warmup_epochs) {
        // Linear warmup from 0 to base_rate
        float warmup_progress = static_cast<float>(epoch) / m_warmup_epochs;
        m_current_rate = m_base_rate * warmup_progress;
    } else if (m_adaptive && validation_loss > 0.0f) {
        // Adaptive dropout based on validation performance
        // If validation loss is increasing, increase dropout slightly
        static float prev_loss = validation_loss;
        if (validation_loss > prev_loss * 1.05f) { // 5% increase threshold
            m_current_rate = std::min(m_current_rate * 1.1f, 0.8f);
        } else if (validation_loss < prev_loss * 0.95f) { // 5% decrease threshold
            m_current_rate = std::max(m_current_rate * 0.9f, m_base_rate * 0.5f);
        }
        prev_loss = validation_loss;
    } else {
        m_current_rate = m_base_rate;
    }
    
    // Clamp to valid range
    m_current_rate = std::clamp(m_current_rate, 0.0f, 0.95f);
}

Tensor AdvancedDropout::apply(const Tensor& input, bool training) {
    if (!training || m_current_rate == 0.0f) {
        return input;
    }
    
    // GPU-accelerated dropout using random mask generation
    Tensor output(input.shape(), input.dtype(), input.device());
    
    // Generate random mask and scale surviving neurons
    std::vector<float> input_data(input.size());
    input.download_data(input_data.data());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float scale = 1.0f / (1.0f - m_current_rate);
    for (size_t i = 0; i < input_data.size(); ++i) {
        if (dis(gen) > m_current_rate) {
            input_data[i] *= scale;
        } else {
            input_data[i] = 0.0f;
        }
    }
    
    output.upload_data(input_data.data());
    return output;
}

// RegularizationManager implementation
void RegularizationManager::add_regularizer(std::unique_ptr<Regularizer> regularizer, float weight) {
    m_regularizers.push_back(std::move(regularizer));
    m_weights.push_back(weight);
}

float RegularizationManager::compute_total_loss(const std::vector<Tensor>& weights) {
    if (!m_enabled || m_regularizers.empty()) {
        return 0.0f;
    }
    
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < m_regularizers.size(); ++i) {
        float reg_loss = m_regularizers[i]->compute_loss(weights);
        total_loss += m_weights[i] * reg_loss;
    }
    
    return total_loss;
}

std::vector<Tensor> RegularizationManager::compute_total_gradients(const std::vector<Tensor>& weights) {
    if (!m_enabled || m_regularizers.empty()) {
        return std::vector<Tensor>();
    }
    
    std::vector<Tensor> total_gradients;
    if (weights.empty()) {
        return total_gradients;
    }
    
    // Initialize with zeros (placeholder)
    total_gradients.reserve(weights.size());
    for (const auto& weight : weights) {
        total_gradients.push_back(weight); // Placeholder - would create zero tensor
    }
    
    // Accumulate gradients from all regularizers
    for (size_t i = 0; i < m_regularizers.size(); ++i) {
        auto reg_gradients = m_regularizers[i]->compute_gradients(weights);
        
        for (size_t j = 0; j < total_gradients.size() && j < reg_gradients.size(); ++j) {
            // TODO: Add weighted regularization gradients
            // total_gradients[j] += m_weights[i] * reg_gradients[j];
        }
    }
    
    return total_gradients;
}

RegularizationManager::RegularizationStats RegularizationManager::get_stats(const std::vector<Tensor>& weights) {
    RegularizationStats stats;
    
    if (!m_enabled || m_regularizers.empty()) {
        return stats;
    }
    
    stats.individual_losses.reserve(m_regularizers.size());
    stats.regularizer_names.reserve(m_regularizers.size());
    
    for (size_t i = 0; i < m_regularizers.size(); ++i) {
        float reg_loss = m_regularizers[i]->compute_loss(weights);
        float weighted_loss = m_weights[i] * reg_loss;
        
        stats.individual_losses.push_back(weighted_loss);
        stats.total_loss += weighted_loss;
        
        // Add regularizer type names (simplified)
        stats.regularizer_names.push_back("Regularizer_" + std::to_string(i));
    }
    
    return stats;
}

// Factory functions
namespace regularization_factory {

std::unique_ptr<RegularizationManager> create_l2_regularization(float lambda) {
    auto manager = std::make_unique<RegularizationManager>();
    manager->add_regularizer(std::make_unique<L2Regularizer>(lambda));
    return manager;
}

std::unique_ptr<RegularizationManager> create_elastic_net(float lambda, float l1_ratio) {
    auto manager = std::make_unique<RegularizationManager>();
    manager->add_regularizer(std::make_unique<ElasticNetRegularizer>(lambda, l1_ratio));
    return manager;
}

std::unique_ptr<RegularizationManager> create_weight_decay(float decay_rate) {
    auto manager = std::make_unique<RegularizationManager>();
    manager->add_regularizer(std::make_unique<WeightDecayRegularizer>(decay_rate));
    return manager;
}

std::unique_ptr<RegularizationManager> create_comprehensive_regularization(
    float l2_lambda, float weight_decay, bool include_l1, float l1_lambda) {
    
    auto manager = std::make_unique<RegularizationManager>();
    
    // Add L2 regularization
    manager->add_regularizer(std::make_unique<L2Regularizer>(l2_lambda));
    
    // Add weight decay
    manager->add_regularizer(std::make_unique<WeightDecayRegularizer>(weight_decay));
    
    // Optionally add L1 regularization
    if (include_l1) {
        manager->add_regularizer(std::make_unique<L1Regularizer>(l1_lambda));
    }
    
    return manager;
}

} // namespace regularization_factory

} // namespace training
} // namespace dlvk
