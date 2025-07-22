#pragma once

#include <memory>
#include <vector>
#include "dlvk/tensor/tensor.h"

namespace dlvk {

class Layer;

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void update(Layer* layer) = 0;
    virtual void set_learning_rate(float lr) = 0;
};

class SGD : public Optimizer {
public:
    SGD(float learning_rate = 0.01f);
    
    void update(Layer* layer) override;
    void set_learning_rate(float lr) override { m_learning_rate = lr; }
    
    float get_learning_rate() const { return m_learning_rate; }

private:
    float m_learning_rate;
};

} // namespace dlvk
