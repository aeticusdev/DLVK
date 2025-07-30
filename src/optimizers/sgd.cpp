#include "dlvk/optimizers/optimizers.h"
#include "dlvk/tensor/tensor_ops.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace dlvk {

SGD::SGD(float learning_rate, float momentum) 
    : m_learning_rate(learning_rate), m_momentum(momentum), m_use_momentum(momentum > 0.0f) {
}

void SGD::update(ModernLayer* layer) {
    if (!layer) return;
    


    layer->update_parameters(*this);
}

void SGD::update_parameter(std::shared_ptr<Tensor>& parameter, 
                          const std::shared_ptr<Tensor>& gradient) {
    if (!parameter || !gradient) {
        return;
    }
    
    auto ops = TensorOps::instance();
    if (!ops) {
        std::cerr << "TensorOps not initialized" << std::endl;
        return;
    }
    

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
    

    if (m_use_momentum && m_momentum > 0.0f) {
        auto param_key = parameter.get();
        if (m_velocity_cache.find(param_key) == m_velocity_cache.end()) {
            m_velocity_cache[param_key] = std::make_shared<Tensor>(parameter->shape(), parameter->dtype(), parameter->device());
            ops->fill(*m_velocity_cache[param_key], 0.0f);
        }
        
        auto velocity = m_velocity_cache[param_key];
        

        auto scaled_velocity = std::make_shared<Tensor>(velocity->shape(), velocity->dtype(), velocity->device());
        ops->scale(*velocity, m_momentum, *scaled_velocity);
        ops->add(*scaled_velocity, *clipped_grad, *velocity);
        

        clipped_grad = velocity;
    }
    

    auto scaled_grad = std::make_shared<Tensor>(parameter->shape(), parameter->dtype(), parameter->device());
    ops->scale(*clipped_grad, -m_learning_rate, *scaled_grad);
    ops->add(*parameter, *scaled_grad, *parameter);
}

} // namespace dlvk
