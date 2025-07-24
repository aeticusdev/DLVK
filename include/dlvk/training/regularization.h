#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <limits>
#include "dlvk/tensor/tensor.h"

namespace dlvk {
namespace training {

/**
 * @brief Regularization types
 */
enum class RegularizationType {
    L1,              // L1 regularization (Lasso)
    L2,              // L2 regularization (Ridge)
    ELASTIC_NET,     // Combination of L1 and L2
    DROPOUT,         // Dropout regularization
    BATCH_NORM,      // Batch normalization
    WEIGHT_DECAY,    // Weight decay (similar to L2 but applied differently)
    SPECTRAL_NORM    // Spectral normalization
};

/**
 * @brief Base regularization interface
 */
class Regularizer {
public:
    virtual ~Regularizer() = default;
    
    /**
     * @brief Compute regularization loss
     */
    virtual float compute_loss(const std::vector<Tensor>& weights) = 0;
    
    /**
     * @brief Compute regularization gradients
     */
    virtual std::vector<Tensor> compute_gradients(const std::vector<Tensor>& weights) = 0;
    
    /**
     * @brief Get regularization strength
     */
    virtual float get_strength() const = 0;
    
    /**
     * @brief Set regularization strength
     */
    virtual void set_strength(float strength) = 0;
};

/**
 * @brief L1 regularization (Lasso)
 * Promotes sparsity in weights
 */
class L1Regularizer : public Regularizer {
private:
    float m_lambda;
    
public:
    L1Regularizer(float lambda = 0.01f) : m_lambda(lambda) {}
    
    float compute_loss(const std::vector<Tensor>& weights) override;
    std::vector<Tensor> compute_gradients(const std::vector<Tensor>& weights) override;
    
    float get_strength() const override { return m_lambda; }
    void set_strength(float strength) override { m_lambda = strength; }
};

/**
 * @brief L2 regularization (Ridge)
 * Promotes smaller weights
 */
class L2Regularizer : public Regularizer {
private:
    float m_lambda;
    
public:
    L2Regularizer(float lambda = 0.01f) : m_lambda(lambda) {}
    
    float compute_loss(const std::vector<Tensor>& weights) override;
    std::vector<Tensor> compute_gradients(const std::vector<Tensor>& weights) override;
    
    float get_strength() const override { return m_lambda; }
    void set_strength(float strength) override { m_lambda = strength; }
};

/**
 * @brief Elastic Net regularization
 * Combination of L1 and L2 regularization
 */
class ElasticNetRegularizer : public Regularizer {
private:
    float m_l1_ratio;   // Mixing parameter between L1 and L2
    float m_lambda;     // Overall regularization strength
    L1Regularizer m_l1_reg;
    L2Regularizer m_l2_reg;
    
public:
    ElasticNetRegularizer(float lambda = 0.01f, float l1_ratio = 0.5f)
        : m_lambda(lambda), m_l1_ratio(l1_ratio),
          m_l1_reg(lambda * l1_ratio), m_l2_reg(lambda * (1.0f - l1_ratio)) {}
    
    float compute_loss(const std::vector<Tensor>& weights) override;
    std::vector<Tensor> compute_gradients(const std::vector<Tensor>& weights) override;
    
    float get_strength() const override { return m_lambda; }
    void set_strength(float strength) override;
    
    float get_l1_ratio() const { return m_l1_ratio; }
    void set_l1_ratio(float ratio);
};

/**
 * @brief Weight decay regularizer
 * Applied directly to optimizer updates
 */
class WeightDecayRegularizer : public Regularizer {
private:
    float m_decay_rate;
    
public:
    WeightDecayRegularizer(float decay_rate = 0.0001f) : m_decay_rate(decay_rate) {}
    
    float compute_loss(const std::vector<Tensor>& weights) override;
    std::vector<Tensor> compute_gradients(const std::vector<Tensor>& weights) override;
    
    float get_strength() const override { return m_decay_rate; }
    void set_strength(float strength) override { m_decay_rate = strength; }
    
    /**
     * @brief Apply weight decay directly to weights
     */
    void apply_decay(std::vector<Tensor>& weights, float learning_rate);
};

/**
 * @brief Advanced dropout with scheduling
 */
class AdvancedDropout {
private:
    float m_base_rate;
    float m_current_rate;
    int m_warmup_epochs;
    int m_current_epoch;
    bool m_adaptive;
    
public:
    AdvancedDropout(float base_rate = 0.5f, 
                   int warmup_epochs = 10,
                   bool adaptive = false)
        : m_base_rate(base_rate), m_current_rate(base_rate),
          m_warmup_epochs(warmup_epochs), m_current_epoch(0),
          m_adaptive(adaptive) {}
    
    /**
     * @brief Update dropout rate based on training progress
     */
    void update_rate(int epoch, float validation_loss = 0.0f);
    
    /**
     * @brief Get current dropout rate
     */
    float get_current_rate() const { return m_current_rate; }
    
    /**
     * @brief Apply dropout to tensor
     */
    Tensor apply(const Tensor& input, bool training = true);
};

/**
 * @brief Regularization manager for coordinating multiple regularizers
 */
class RegularizationManager {
private:
    std::vector<std::unique_ptr<Regularizer>> m_regularizers;
    std::vector<float> m_weights;  // Relative weights for each regularizer
    bool m_enabled;
    
public:
    RegularizationManager() : m_enabled(true) {}
    
    /**
     * @brief Add regularizer with optional weight
     */
    void add_regularizer(std::unique_ptr<Regularizer> regularizer, float weight = 1.0f);
    
    /**
     * @brief Compute total regularization loss
     */
    float compute_total_loss(const std::vector<Tensor>& weights);
    
    /**
     * @brief Compute total regularization gradients
     */
    std::vector<Tensor> compute_total_gradients(const std::vector<Tensor>& weights);
    
    /**
     * @brief Enable/disable all regularization
     */
    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }
    
    /**
     * @brief Get regularization statistics
     */
    struct RegularizationStats {
        float total_loss = 0.0f;
        std::vector<float> individual_losses;
        std::vector<std::string> regularizer_names;
    };
    RegularizationStats get_stats(const std::vector<Tensor>& weights);
    
    /**
     * @brief Clear all regularizers
     */
    void clear() { m_regularizers.clear(); m_weights.clear(); }
    
    /**
     * @brief Get number of regularizers
     */
    size_t size() const { return m_regularizers.size(); }
};

/**
 * @brief Factory functions for common regularization setups
 */
namespace regularization_factory {
    
    /**
     * @brief Create standard L2 regularization
     */
    std::unique_ptr<RegularizationManager> create_l2_regularization(float lambda = 0.01f);
    
    /**
     * @brief Create L1 + L2 combination
     */
    std::unique_ptr<RegularizationManager> create_elastic_net(float lambda = 0.01f, float l1_ratio = 0.5f);
    
    /**
     * @brief Create weight decay regularization
     */
    std::unique_ptr<RegularizationManager> create_weight_decay(float decay_rate = 0.0001f);
    
    /**
     * @brief Create comprehensive regularization suite
     */
    std::unique_ptr<RegularizationManager> create_comprehensive_regularization(
        float l2_lambda = 0.01f,
        float weight_decay = 0.0001f,
        bool include_l1 = false,
        float l1_lambda = 0.001f
    );
}

} // namespace training
} // namespace dlvk
