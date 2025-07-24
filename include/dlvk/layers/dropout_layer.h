#pragma once

#include "layer.h"
#include "../tensor/tensor.h"
#include <memory>
#include <random>

namespace dlvk {

class DropoutLayer : public Layer {
private:
    float dropout_rate_;
    bool training_;
    std::shared_ptr<Tensor> mask_;  // Dropout mask for backward pass
    VulkanDevice& device_;
    std::mt19937 generator_;
    std::uniform_real_distribution<float> distribution_;

public:
    DropoutLayer(VulkanDevice& device, float dropout_rate = 0.5f);
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    void update_weights([[maybe_unused]] float learning_rate) override {} // No weights to update
    std::unique_ptr<Layer> clone() const override;
    
    void set_training(bool training) { training_ = training; }
    bool is_training() const { return training_; }
    float get_dropout_rate() const { return dropout_rate_; }
    void set_dropout_rate(float rate) { dropout_rate_ = rate; }
};

} // namespace dlvk
