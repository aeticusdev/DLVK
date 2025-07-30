#include "dlvk/optimizers/optimizers.h"
#include "dlvk/tensor/tensor_ops.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace dlvk {

Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
    : m_learning_rate(learning_rate), m_beta1(beta1), m_beta2(beta2), 
      m_epsilon(epsilon), m_step_count(0) {
}

void Adam::update(ModernLayer* layer) {
    if (!layer) return;
    

    m_step_count++;
    

    layer->update_parameters(*this);
}

void Adam::update_parameter(std::shared_ptr<Tensor>& parameter, 
                           const std::shared_ptr<Tensor>& gradient) {
    if (!parameter || !gradient) {
        return;
    }
    
    auto ops = TensorOps::instance();
    if (!ops) {
        std::cerr << "TensorOps not initialized" << std::endl;
        return;
    }
    

    m_step_count++;
    

    auto clipped_grad = std::make_shared<Tensor>(parameter->shape(), parameter->dtype(), parameter->device());
    ops->copy(*gradient, *clipped_grad);
    

    if (m_use_grad_clip_norm) {
        if (!ops->gradient_clip_by_norm(*gradient, m_grad_clip_norm, *clipped_grad)) {
            std::cerr << "Failed to apply gradient norm clipping" << std::endl;
            return;
        }
    }
    
    if (m_use_grad_clip_value) {
        if (!ops->gradient_clip_by_value(*clipped_grad, m_grad_clip_min, m_grad_clip_max, *clipped_grad)) {
            std::cerr << "Failed to apply gradient value clipping" << std::endl;
            return;
        }
    }
    

    auto momentum_it = m_momentum_cache.find(parameter.get());
    std::shared_ptr<Tensor> momentum;
    
    if (momentum_it == m_momentum_cache.end()) {
        momentum = std::make_shared<Tensor>(parameter->shape(), parameter->dtype(), parameter->device());
        ops->fill(*momentum, 0.0f);
        m_momentum_cache[parameter.get()] = momentum;
    } else {
        momentum = momentum_it->second;
    }
    

    auto velocity_it = m_velocity_cache.find(parameter.get());
    std::shared_ptr<Tensor> velocity;
    
    if (velocity_it == m_velocity_cache.end()) {
        velocity = std::make_shared<Tensor>(parameter->shape(), parameter->dtype(), parameter->device());
        ops->fill(*velocity, 0.0f);
        m_velocity_cache[parameter.get()] = velocity;
    } else {
        velocity = velocity_it->second;
    }
    

    auto new_momentum = std::make_shared<Tensor>(parameter->shape(), parameter->dtype(), parameter->device());
    auto new_velocity = std::make_shared<Tensor>(parameter->shape(), parameter->dtype(), parameter->device());
    

    if (!ops->adam_update(*clipped_grad, *momentum, *velocity, 
                         *parameter, *new_momentum, *new_velocity,
                         m_learning_rate, m_beta1, m_beta2, m_epsilon)) {
        std::cerr << "Failed to perform Adam GPU update" << std::endl;
        return;
    }
    

    ops->copy(*new_momentum, *momentum);
    ops->copy(*new_velocity, *velocity);
}

} // namespace dlvk
