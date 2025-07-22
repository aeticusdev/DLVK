#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "dlvk/tensor/tensor.h"

namespace dlvk {

class Layer;

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void update(Layer* layer) = 0;
    virtual void set_learning_rate(float lr) = 0;
    
    // New parameter-based update methods for advanced optimizers
    virtual void update_parameter(std::shared_ptr<Tensor>& parameter, 
                                 const std::shared_ptr<Tensor>& gradient) {}
    virtual void step() {}
    virtual void zero_gradients() {}
};

class SGD : public Optimizer {
public:
    SGD(float learning_rate = 0.01f, float momentum = 0.0f);
    
    void update(Layer* layer) override;
    void set_learning_rate(float lr) override { m_learning_rate = lr; }
    
    void update_parameter(std::shared_ptr<Tensor>& parameter, 
                         const std::shared_ptr<Tensor>& gradient) override;
    
    float get_learning_rate() const { return m_learning_rate; }

private:
    float m_learning_rate;
    float m_momentum;
    bool m_use_momentum;
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m_velocity_cache;
};

class Adam : public Optimizer {
public:
    Adam(float learning_rate = 0.001f, 
         float beta1 = 0.9f, 
         float beta2 = 0.999f, 
         float epsilon = 1e-8f);
    
    void update(Layer* layer) override;
    void set_learning_rate(float lr) override { m_learning_rate = lr; }
    
    void update_parameter(std::shared_ptr<Tensor>& parameter, 
                         const std::shared_ptr<Tensor>& gradient) override;
    
    void step() override { m_step_count++; }
    
    float get_learning_rate() const { return m_learning_rate; }

private:
    float m_learning_rate;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    size_t m_step_count;
    
    // Momentum and velocity caches for each parameter
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m_momentum_cache;
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m_velocity_cache;
};

class RMSprop : public Optimizer {
public:
    RMSprop(float learning_rate = 0.01f, 
            float alpha = 0.99f, 
            float epsilon = 1e-8f);
    
    void update(Layer* layer) override;
    void set_learning_rate(float lr) override { m_learning_rate = lr; }
    
    void update_parameter(std::shared_ptr<Tensor>& parameter, 
                         const std::shared_ptr<Tensor>& gradient) override;
    
    float get_learning_rate() const { return m_learning_rate; }

private:
    float m_learning_rate;
    float m_alpha;
    float m_epsilon;
    
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m_square_avg_cache;
};

} // namespace dlvk
