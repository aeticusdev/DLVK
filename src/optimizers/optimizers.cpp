#include "dlvk/optimizers/optimizers.h"
#include <cmath>

namespace dlvk {

// SGD Implementation
SGD::SGD(float learning_rate, float momentum) 
    : m_learning_rate(learning_rate), m_momentum(momentum), 
      m_use_momentum(momentum > 0.0f) {}

void SGD::update(ModernLayer* layer) {
    if (layer) {
        layer->update_parameters(*this);
    }
}

void SGD::update_parameter(std::shared_ptr<Tensor>& parameter, 
                          const std::shared_ptr<Tensor>& gradient) {
    // Apply gradient clipping if enabled
    auto clipped_gradient = gradient;
    if (m_use_grad_clip_norm || m_use_grad_clip_value) {
        std::vector<std::shared_ptr<Tensor>> grad_vec = {gradient};
        
        if (m_use_grad_clip_norm) {
            GradientClipping::clip_grad_norm(grad_vec, m_grad_clip_norm);
        }
        if (m_use_grad_clip_value) {
            GradientClipping::clip_grad_value(grad_vec, m_grad_clip_min, m_grad_clip_max);
        }
        
        clipped_gradient = grad_vec[0];
    }
    
    if (m_use_momentum) {
        // v = momentum * v + gradient
        // parameter = parameter - learning_rate * v
        
        auto it = m_velocity_cache.find(parameter.get());
        if (it == m_velocity_cache.end()) {
            // Initialize velocity
            m_velocity_cache[parameter.get()] = std::make_shared<Tensor>(
                parameter->shape(), DataType::FLOAT32, parameter->device());
            
            std::vector<float> zero_data(parameter->size(), 0.0f);
            m_velocity_cache[parameter.get()]->upload_data(zero_data.data());
        }
        
        auto velocity = m_velocity_cache[parameter.get()];
        
        // Update velocity: v = momentum * v + clipped_gradient
        auto momentum_v = velocity->multiply_scalar(m_momentum);
        auto new_velocity = momentum_v->add(*clipped_gradient);
        
        // Copy back to velocity cache
        std::vector<float> velocity_data(new_velocity->size());
        new_velocity->download_data(velocity_data.data());
        velocity->upload_data(velocity_data.data());
        
        // Update parameter: param = param - lr * velocity
        auto update = new_velocity->multiply_scalar(m_learning_rate);
        auto new_param = parameter->subtract(*update);
        
        // Copy back to parameter
        std::vector<float> param_data(new_param->size());
        new_param->download_data(param_data.data());
        parameter->upload_data(param_data.data());
        
    } else {
        // Simple SGD: parameter = parameter - learning_rate * clipped_gradient
        auto update = clipped_gradient->multiply_scalar(m_learning_rate);
        auto new_param = parameter->subtract(*update);
        
        std::vector<float> param_data(new_param->size());
        new_param->download_data(param_data.data());
        parameter->upload_data(param_data.data());
    }
}

// Adam Implementation
Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
    : m_learning_rate(learning_rate), m_beta1(beta1), m_beta2(beta2), 
      m_epsilon(epsilon), m_step_count(0) {}

void Adam::update(ModernLayer* layer) {
    if (layer) {
        layer->update_parameters(*this);
    }
}

void Adam::update_parameter(std::shared_ptr<Tensor>& parameter, 
                           const std::shared_ptr<Tensor>& gradient) {
    // Apply gradient clipping if enabled
    auto clipped_gradient = gradient;
    if (m_use_grad_clip_norm || m_use_grad_clip_value) {
        std::vector<std::shared_ptr<Tensor>> grad_vec = {gradient};
        
        if (m_use_grad_clip_norm) {
            GradientClipping::clip_grad_norm(grad_vec, m_grad_clip_norm);
        }
        if (m_use_grad_clip_value) {
            GradientClipping::clip_grad_value(grad_vec, m_grad_clip_min, m_grad_clip_max);
        }
        
        clipped_gradient = grad_vec[0];
    }
    
    // Initialize momentum and velocity if needed
    auto param_ptr = parameter.get();
    
    if (m_momentum_cache.find(param_ptr) == m_momentum_cache.end()) {
        m_momentum_cache[param_ptr] = std::make_shared<Tensor>(
            parameter->shape(), DataType::FLOAT32, parameter->device());
        std::vector<float> zero_data(parameter->size(), 0.0f);
        m_momentum_cache[param_ptr]->upload_data(zero_data.data());
    }
    
    if (m_velocity_cache.find(param_ptr) == m_velocity_cache.end()) {
        m_velocity_cache[param_ptr] = std::make_shared<Tensor>(
            parameter->shape(), DataType::FLOAT32, parameter->device());
        std::vector<float> zero_data(parameter->size(), 0.0f);
        m_velocity_cache[param_ptr]->upload_data(zero_data.data());
    }
    
    auto momentum = m_momentum_cache[param_ptr];
    auto velocity = m_velocity_cache[param_ptr];
    
    // m = beta1 * m + (1 - beta1) * clipped_gradient
    auto beta1_m = momentum->multiply_scalar(m_beta1);
    auto grad_term = clipped_gradient->multiply_scalar(1.0f - m_beta1);
    auto new_momentum = beta1_m->add(*grad_term);
    
    // v = beta2 * v + (1 - beta2) * clipped_gradient^2
    auto grad_squared = clipped_gradient->multiply(*clipped_gradient);
    auto beta2_v = velocity->multiply_scalar(m_beta2);
    auto grad_sq_term = grad_squared->multiply_scalar(1.0f - m_beta2);
    auto new_velocity = beta2_v->add(*grad_sq_term);
    
    // Bias correction
    float m_hat_correction = 1.0f / (1.0f - std::pow(m_beta1, m_step_count + 1));
    float v_hat_correction = 1.0f / (1.0f - std::pow(m_beta2, m_step_count + 1));
    
    auto m_hat = new_momentum->multiply_scalar(m_hat_correction);
    auto v_hat = new_velocity->multiply_scalar(v_hat_correction);
    
    // Download v_hat, compute sqrt, and create denominator
    std::vector<float> v_hat_data(v_hat->size());
    v_hat->download_data(v_hat_data.data());
    
    std::vector<float> denominator_data(v_hat_data.size());
    for (size_t i = 0; i < v_hat_data.size(); ++i) {
        denominator_data[i] = std::sqrt(v_hat_data[i]) + m_epsilon;
    }
    
    auto denominator = std::make_shared<Tensor>(
        parameter->shape(), DataType::FLOAT32, parameter->device());
    denominator->upload_data(denominator_data.data());
    
    // Update: param = param - lr * m_hat / denominator
    auto update_numerator = m_hat->multiply_scalar(m_learning_rate);
    auto update = update_numerator->divide(*denominator);
    auto new_param = parameter->subtract(*update);
    
    // Copy results back
    std::vector<float> momentum_data(new_momentum->size());
    std::vector<float> velocity_data(new_velocity->size());
    std::vector<float> param_data(new_param->size());
    
    new_momentum->download_data(momentum_data.data());
    new_velocity->download_data(velocity_data.data());
    new_param->download_data(param_data.data());
    
    momentum->upload_data(momentum_data.data());
    velocity->upload_data(velocity_data.data());
    parameter->upload_data(param_data.data());
}

// RMSprop Implementation
RMSprop::RMSprop(float learning_rate, float alpha, float epsilon)
    : m_learning_rate(learning_rate), m_alpha(alpha), m_epsilon(epsilon) {}

void RMSprop::update(ModernLayer* layer) {
    if (layer) {
        layer->update_parameters(*this);
    }
}

void RMSprop::update_parameter(std::shared_ptr<Tensor>& parameter, 
                              const std::shared_ptr<Tensor>& gradient) {
    // Apply gradient clipping if enabled
    auto clipped_gradient = gradient;
    if (m_use_grad_clip_norm || m_use_grad_clip_value) {
        std::vector<std::shared_ptr<Tensor>> grad_vec = {gradient};
        
        if (m_use_grad_clip_norm) {
            GradientClipping::clip_grad_norm(grad_vec, m_grad_clip_norm);
        }
        if (m_use_grad_clip_value) {
            GradientClipping::clip_grad_value(grad_vec, m_grad_clip_min, m_grad_clip_max);
        }
        
        clipped_gradient = grad_vec[0];
    }
    
    auto param_ptr = parameter.get();
    
    // Initialize square average if needed
    if (m_square_avg_cache.find(param_ptr) == m_square_avg_cache.end()) {
        m_square_avg_cache[param_ptr] = std::make_shared<Tensor>(
            parameter->shape(), DataType::FLOAT32, parameter->device());
        std::vector<float> zero_data(parameter->size(), 0.0f);
        m_square_avg_cache[param_ptr]->upload_data(zero_data.data());
    }
    
    auto square_avg = m_square_avg_cache[param_ptr];
    
    // square_avg = alpha * square_avg + (1 - alpha) * clipped_gradient^2
    auto grad_squared = clipped_gradient->multiply(*clipped_gradient);
    auto alpha_avg = square_avg->multiply_scalar(m_alpha);
    auto grad_term = grad_squared->multiply_scalar(1.0f - m_alpha);
    auto new_square_avg = alpha_avg->add(*grad_term);
    
    // Download square_avg, compute sqrt + epsilon
    std::vector<float> square_avg_data(new_square_avg->size());
    new_square_avg->download_data(square_avg_data.data());
    
    std::vector<float> denominator_data(square_avg_data.size());
    for (size_t i = 0; i < square_avg_data.size(); ++i) {
        denominator_data[i] = std::sqrt(square_avg_data[i]) + m_epsilon;
    }
    
    auto denominator = std::make_shared<Tensor>(
        parameter->shape(), DataType::FLOAT32, parameter->device());
    denominator->upload_data(denominator_data.data());
    
    // Update: param = param - lr * clipped_gradient / denominator
    auto update_numerator = clipped_gradient->multiply_scalar(m_learning_rate);
    auto update = update_numerator->divide(*denominator);
    auto new_param = parameter->subtract(*update);
    
    // Copy results back
    std::vector<float> avg_data(new_square_avg->size());
    std::vector<float> param_data(new_param->size());
    
    new_square_avg->download_data(avg_data.data());
    new_param->download_data(param_data.data());
    
    square_avg->upload_data(avg_data.data());
    parameter->upload_data(param_data.data());
}

// Gradient clipping implementations
namespace GradientClipping {

float compute_grad_norm(const std::vector<std::shared_ptr<Tensor>>& gradients) {
    float total_norm_squared = 0.0f;
    
    for (const auto& grad : gradients) {
        if (!grad) continue;
        
        std::vector<float> grad_data(grad->size());
        grad->download_data(grad_data.data());
        
        for (float val : grad_data) {
            total_norm_squared += val * val;
        }
    }
    
    return std::sqrt(total_norm_squared);
}

void clip_grad_norm(std::vector<std::shared_ptr<Tensor>>& gradients, float max_norm) {
    if (gradients.empty()) return;
    
    float current_norm = compute_grad_norm(gradients);
    
    if (current_norm <= max_norm) {
        return; // No clipping needed
    }
    
    float clip_factor = max_norm / current_norm;
    
    // Scale all gradients by the clip factor
    for (auto& grad : gradients) {
        if (!grad) continue;
        
        auto clipped_grad = grad->multiply_scalar(clip_factor);
        
        std::vector<float> clipped_data(clipped_grad->size());
        clipped_grad->download_data(clipped_data.data());
        grad->upload_data(clipped_data.data());
    }
}

void clip_grad_value(std::vector<std::shared_ptr<Tensor>>& gradients, 
                     float min_value, float max_value) {
    for (auto& grad : gradients) {
        if (!grad) continue;
        
        std::vector<float> grad_data(grad->size());
        grad->download_data(grad_data.data());
        
        // Clip each gradient value
        for (float& val : grad_data) {
            val = std::max(min_value, std::min(max_value, val));
        }
        
        grad->upload_data(grad_data.data());
    }
}

} // namespace GradientClipping

} // namespace dlvk
