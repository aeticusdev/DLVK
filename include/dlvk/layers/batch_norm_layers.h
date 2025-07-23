#pragma once

#include "layer.h"
#include "../tensor/tensor.h"
#include <memory>

namespace dlvk {

class BatchNorm1DLayer : public Layer {
private:
    std::shared_ptr<Tensor> gamma_;        // Scale parameter [features]
    std::shared_ptr<Tensor> beta_;         // Shift parameter [features]
    std::shared_ptr<Tensor> running_mean_; // Running mean [features]
    std::shared_ptr<Tensor> running_var_;  // Running variance [features]
    std::shared_ptr<Tensor> last_input_;   // For backward pass
    std::shared_ptr<Tensor> last_normalized_; // For backward pass
    
    size_t num_features_;
    float momentum_;
    float epsilon_;
    bool training_;
    VulkanDevice& device_;

public:
    BatchNorm1DLayer(VulkanDevice& device, size_t num_features,
                     float momentum = 0.1f, float epsilon = 1e-5f);
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    void update_weights(float learning_rate) override;
    
    void set_training(bool training) { training_ = training; }
    bool is_training() const { return training_; }
    
    void initialize_parameters();
};

class BatchNorm2DLayer : public Layer {
private:
    std::shared_ptr<Tensor> gamma_;        // Scale parameter [channels]
    std::shared_ptr<Tensor> beta_;         // Shift parameter [channels]
    std::shared_ptr<Tensor> running_mean_; // Running mean [channels]
    std::shared_ptr<Tensor> running_var_;  // Running variance [channels]
    std::shared_ptr<Tensor> last_input_;   // For backward pass
    std::shared_ptr<Tensor> last_normalized_; // For backward pass
    
    size_t num_channels_;
    float momentum_;
    float epsilon_;
    bool training_;
    VulkanDevice& device_;

public:
    BatchNorm2DLayer(VulkanDevice& device, size_t num_channels,
                     float momentum = 0.1f, float epsilon = 1e-5f);
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    void update_weights(float learning_rate) override;
    
    void set_training(bool training) { training_ = training; }
    bool is_training() const { return training_; }
    
    void initialize_parameters();
};

} // namespace dlvk
