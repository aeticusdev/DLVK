#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "dlvk/tensor/tensor.h"
#include "dlvk/layers/modern_layer.h"

namespace dlvk {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void update(ModernLayer* layer) = 0;
    virtual void set_learning_rate(float lr) = 0;
    virtual float get_learning_rate() const = 0;
    

    virtual void set_lr(float lr) { set_learning_rate(lr); }
    virtual float get_lr() const { return get_learning_rate(); }
    

    virtual void update_parameter(std::shared_ptr<Tensor>& /*parameter*/, 
                                 const std::shared_ptr<Tensor>& /*gradient*/) {}
    virtual void step() {}
    virtual void zero_gradients() {}
};

class SGD : public Optimizer {
public:
    SGD(float learning_rate = 0.01f, float momentum = 0.0f);
    
    void update(ModernLayer* layer) override;
    void set_learning_rate(float lr) override { m_learning_rate = lr; }
    
    void update_parameter(std::shared_ptr<Tensor>& parameter, 
                         const std::shared_ptr<Tensor>& gradient) override;
    

    void set_grad_clip_norm(float max_norm) { m_grad_clip_norm = max_norm; m_use_grad_clip_norm = true; }
    void set_grad_clip_value(float min_val, float max_val) { 
        m_grad_clip_min = min_val; m_grad_clip_max = max_val; m_use_grad_clip_value = true; 
    }
    void disable_grad_clipping() { m_use_grad_clip_norm = false; m_use_grad_clip_value = false; }
    
    float get_learning_rate() const { return m_learning_rate; }

private:
    float m_learning_rate;
    float m_momentum;
    bool m_use_momentum;
    

    bool m_use_grad_clip_norm = false;
    bool m_use_grad_clip_value = false;
    float m_grad_clip_norm = 1.0f;
    float m_grad_clip_min = -1.0f;
    float m_grad_clip_max = 1.0f;
    
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m_velocity_cache;
};

class Adam : public Optimizer {
public:
    Adam(float learning_rate = 0.001f, 
         float beta1 = 0.9f, 
         float beta2 = 0.999f, 
         float epsilon = 1e-8f);
    
    void update(ModernLayer* layer) override;
    void set_learning_rate(float lr) override { m_learning_rate = lr; }
    
    void update_parameter(std::shared_ptr<Tensor>& parameter, 
                         const std::shared_ptr<Tensor>& gradient) override;
    
    void step() override { m_step_count++; }
    

    void set_grad_clip_norm(float max_norm) { m_grad_clip_norm = max_norm; m_use_grad_clip_norm = true; }
    void set_grad_clip_value(float min_val, float max_val) { 
        m_grad_clip_min = min_val; m_grad_clip_max = max_val; m_use_grad_clip_value = true; 
    }
    void disable_grad_clipping() { m_use_grad_clip_norm = false; m_use_grad_clip_value = false; }
    
    float get_learning_rate() const { return m_learning_rate; }

private:
    float m_learning_rate;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    size_t m_step_count;
    

    bool m_use_grad_clip_norm = false;
    bool m_use_grad_clip_value = false;
    float m_grad_clip_norm = 1.0f;
    float m_grad_clip_min = -1.0f;
    float m_grad_clip_max = 1.0f;
    

    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m_momentum_cache;
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m_velocity_cache;
};

class RMSprop : public Optimizer {
public:
    RMSprop(float learning_rate = 0.01f, 
            float alpha = 0.99f, 
            float epsilon = 1e-8f);
    
    void update(ModernLayer* layer) override;
    void set_learning_rate(float lr) override { m_learning_rate = lr; }
    
    void update_parameter(std::shared_ptr<Tensor>& parameter, 
                         const std::shared_ptr<Tensor>& gradient) override;
    

    void set_grad_clip_norm(float max_norm) { m_grad_clip_norm = max_norm; m_use_grad_clip_norm = true; }
    void set_grad_clip_value(float min_val, float max_val) { 
        m_grad_clip_min = min_val; m_grad_clip_max = max_val; m_use_grad_clip_value = true; 
    }
    void disable_grad_clipping() { m_use_grad_clip_norm = false; m_use_grad_clip_value = false; }
    
    float get_learning_rate() const { return m_learning_rate; }

private:
    float m_learning_rate;
    float m_alpha;
    float m_epsilon;
    

    bool m_use_grad_clip_norm = false;
    bool m_use_grad_clip_value = false;
    float m_grad_clip_norm = 1.0f;
    float m_grad_clip_min = -1.0f;
    float m_grad_clip_max = 1.0f;
    
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m_square_avg_cache;
};


namespace GradientClipping {

    void clip_grad_norm(std::vector<std::shared_ptr<Tensor>>& gradients, 
                        float max_norm);
    

    void clip_grad_value(std::vector<std::shared_ptr<Tensor>>& gradients, 
                         float min_value, float max_value);
                         

    float compute_grad_norm(const std::vector<std::shared_ptr<Tensor>>& gradients);
}

} // namespace dlvk
