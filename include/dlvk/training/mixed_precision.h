#pragma once

#include <memory>
#include <cmath>
#include <functional>
#include "dlvk/tensor/tensor.h"
#include "dlvk/optimizers/optimizers.h"

namespace dlvk {
namespace training {

/**
 * @brief Precision modes for mixed precision training
 */
enum class PrecisionMode {
    FP32,    // Full precision (default)
    FP16,    // Half precision
    MIXED    // Mixed precision (FP16 forward, FP32 backward)
};

/**
 * @brief Gradient scaler for mixed precision training
 * Prevents gradient underflow in FP16 computations
 */
class GradientScaler {
private:
    float m_scale;
    float m_scale_factor;
    int m_scale_window;
    int m_unskipped_count;
    float m_backoff_factor;
    float m_growth_factor;
    bool m_enabled;

public:
    GradientScaler(float init_scale = 65536.0f,  // 2^16
                   float scale_factor = 2.0f,
                   int scale_window = 2000,
                   float backoff_factor = 0.5f,
                   float growth_factor = 2.0f,
                   bool enabled = true)
        : m_scale(init_scale), m_scale_factor(scale_factor), 
          m_scale_window(scale_window), m_unskipped_count(0),
          m_backoff_factor(backoff_factor), m_growth_factor(growth_factor),
          m_enabled(enabled) {}

    /**
     * @brief Scale loss for backward pass
     */
    Tensor scale_loss(const Tensor& loss);

    /**
     * @brief Unscale gradients before optimizer step
     */
    void unscale_gradients(std::shared_ptr<Optimizer> optimizer);

    /**
     * @brief Check for gradient overflow/underflow
     */
    bool has_overflow() const;

    /**
     * @brief Update scaling factor based on gradient status
     */
    void update();

    /**
     * @brief Get current scale factor
     */
    float get_scale() const { return m_enabled ? m_scale : 1.0f; }

    /**
     * @brief Enable/disable gradient scaling
     */
    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }
};

/**
 * @brief Mixed precision training context
 */
class MixedPrecisionContext {
private:
    PrecisionMode m_mode;
    std::unique_ptr<GradientScaler> m_scaler;
    bool m_autocast_enabled;

public:
    MixedPrecisionContext(PrecisionMode mode = PrecisionMode::MIXED,
                         bool enable_scaler = true,
                         bool autocast = true)
        : m_mode(mode), m_autocast_enabled(autocast) {
        if (enable_scaler && mode != PrecisionMode::FP32) {
            m_scaler = std::make_unique<GradientScaler>();
        }
    }

    /**
     * @brief Get precision mode
     */
    PrecisionMode get_mode() const { return m_mode; }

    /**
     * @brief Get gradient scaler
     */
    GradientScaler* get_scaler() const { return m_scaler.get(); }

    /**
     * @brief Check if autocast is enabled
     */
    bool is_autocast_enabled() const { return m_autocast_enabled; }

    /**
     * @brief Convert tensor to forward pass precision
     */
    Tensor to_forward_precision(const Tensor& input);

    /**
     * @brief Convert tensor to backward pass precision
     */
    Tensor to_backward_precision(const Tensor& input);

    /**
     * @brief Estimate memory savings
     */
    float estimate_memory_savings() const;
};

/**
 * @brief RAII autocast context for automatic precision conversion
 */
class AutocastContext {
private:
    static thread_local bool s_enabled;
    static thread_local PrecisionMode s_mode;
    bool m_prev_enabled;
    PrecisionMode m_prev_mode;

public:
    AutocastContext(bool enabled, PrecisionMode mode = PrecisionMode::FP16);
    ~AutocastContext();

    static bool is_enabled() { return s_enabled; }
    static PrecisionMode get_mode() { return s_mode; }
};

/**
 * @brief Mixed precision trainer wrapper
 */
class MixedPrecisionTrainer {
private:
    std::unique_ptr<MixedPrecisionContext> m_context;
    
public:
    MixedPrecisionTrainer(PrecisionMode mode = PrecisionMode::MIXED)
        : m_context(std::make_unique<MixedPrecisionContext>(mode)) {}

    /**
     * @brief Get mixed precision context
     */
    MixedPrecisionContext& get_context() { return *m_context; }
    const MixedPrecisionContext& get_context() const { return *m_context; }

    /**
     * @brief Perform forward pass with automatic precision
     */
    Tensor forward_with_autocast(std::function<Tensor()> forward_fn);

    /**
     * @brief Perform backward pass with gradient scaling
     */
    void backward_with_scaling(const Tensor& loss, std::shared_ptr<Optimizer> optimizer);

    /**
     * @brief Get memory usage statistics
     */
    struct MemoryStats {
        size_t fp32_memory = 0;
        size_t fp16_memory = 0;
        float savings_ratio = 0.0f;
    };
    MemoryStats get_memory_stats() const;
};

} // namespace training
} // namespace dlvk
