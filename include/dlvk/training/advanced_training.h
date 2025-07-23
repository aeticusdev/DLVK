#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <chrono>
#include "dlvk/training/trainer.h"
#include "dlvk/training/mixed_precision.h"
#include "dlvk/training/regularization.h"
#include "dlvk/data/dataloader.h"

namespace dlvk {
namespace training {

/**
 * @brief Learning rate scheduling strategies
 */
enum class LRScheduleType {
    CONSTANT,        // No scheduling
    STEP_DECAY,      // Step decay at fixed intervals
    EXPONENTIAL,     // Exponential decay
    COSINE_ANNEALING,// Cosine annealing
    REDUCE_ON_PLATEAU,// Reduce when validation stops improving
    CYCLIC,          // Cyclic learning rates
    ONE_CYCLE        // One cycle policy
};

/**
 * @brief Learning rate scheduler base class
 */
class LearningRateScheduler {
public:
    virtual ~LearningRateScheduler() = default;
    
    /**
     * @brief Get learning rate for current step/epoch
     */
    virtual float get_lr(int step, float current_lr, float validation_metric = 0.0f) = 0;
    
    /**
     * @brief Update scheduler state
     */
    virtual void step(float validation_metric = 0.0f) {}
    
    /**
     * @brief Reset scheduler to initial state
     */
    virtual void reset() {}
};

/**
 * @brief Cosine annealing scheduler
 */
class CosineAnnealingScheduler : public LearningRateScheduler {
private:
    float m_initial_lr;
    float m_min_lr;
    int m_max_steps;
    int m_current_step;
    
public:
    CosineAnnealingScheduler(float initial_lr, float min_lr, int max_steps)
        : m_initial_lr(initial_lr), m_min_lr(min_lr), 
          m_max_steps(max_steps), m_current_step(0) {}
    
    float get_lr(int step, float current_lr, float validation_metric = 0.0f) override;
    void step(float validation_metric = 0.0f) override { m_current_step++; }
    void reset() override { m_current_step = 0; }
};

/**
 * @brief One cycle learning rate policy
 */
class OneCycleScheduler : public LearningRateScheduler {
private:
    float m_max_lr;
    float m_initial_lr;
    float m_final_lr;
    int m_total_steps;
    int m_pct_start;  // Percentage of cycle spent increasing LR
    int m_current_step;
    
public:
    OneCycleScheduler(float max_lr, int total_steps, float initial_lr_ratio = 0.1f,
                     float final_lr_ratio = 0.01f, int pct_start = 30)
        : m_max_lr(max_lr), m_total_steps(total_steps), 
          m_pct_start(pct_start), m_current_step(0) {
        m_initial_lr = max_lr * initial_lr_ratio;
        m_final_lr = max_lr * final_lr_ratio;
    }
    
    float get_lr(int step, float current_lr, float validation_metric = 0.0f) override;
    void step(float validation_metric = 0.0f) override { m_current_step++; }
    void reset() override { m_current_step = 0; }
};

/**
 * @brief Reduce on plateau scheduler
 */
class ReduceOnPlateauScheduler : public LearningRateScheduler {
private:
    float m_factor;
    int m_patience;
    float m_threshold;
    int m_cooldown;
    float m_min_lr;
    
    float m_best_metric;
    int m_wait_count;
    int m_cooldown_count;
    bool m_mode_max;  // true for maximizing metric (accuracy), false for minimizing (loss)
    
public:
    ReduceOnPlateauScheduler(float factor = 0.1f, int patience = 10, 
                           float threshold = 1e-4f, int cooldown = 0,
                           float min_lr = 0.0f, bool mode_max = false)
        : m_factor(factor), m_patience(patience), m_threshold(threshold),
          m_cooldown(cooldown), m_min_lr(min_lr), m_mode_max(mode_max),
          m_wait_count(0), m_cooldown_count(0) {
        m_best_metric = mode_max ? -std::numeric_limits<float>::infinity() 
                                 : std::numeric_limits<float>::infinity();
    }
    
    float get_lr(int step, float current_lr, float validation_metric = 0.0f) override;
    void step(float validation_metric = 0.0f) override;
    void reset() override;
};

/**
 * @brief Advanced training configuration
 */
struct AdvancedTrainingConfig {
    // Mixed precision
    bool use_mixed_precision = false;
    PrecisionMode precision_mode = PrecisionMode::MIXED;
    bool enable_gradient_scaling = true;
    
    // Regularization
    bool use_regularization = false;
    float l2_lambda = 0.01f;
    float weight_decay = 0.0001f;
    bool use_dropout_scheduling = false;
    
    // Learning rate scheduling
    LRScheduleType lr_schedule = LRScheduleType::CONSTANT;
    float lr_decay_factor = 0.1f;
    int lr_decay_patience = 10;
    float lr_min = 1e-7f;
    
    // Gradient management
    bool use_gradient_clipping = false;
    float gradient_clip_value = 1.0f;
    bool gradient_clip_by_norm = true;
    
    // Training enhancements
    bool use_early_stopping = true;
    int early_stopping_patience = 15;
    bool save_best_model = true;
    std::string checkpoint_dir = "./checkpoints";
    
    // Validation
    float validation_split = 0.2f;
    int validation_frequency = 1;  // Every N epochs
    
    // Logging and monitoring
    bool verbose_training = true;
    int log_frequency = 10;  // Every N batches
    bool save_training_curves = true;
};

/**
 * @brief Advanced trainer with comprehensive features
 */
class AdvancedTrainer {
private:
    std::shared_ptr<Model> m_model;
    std::shared_ptr<Optimizer> m_optimizer;
    std::shared_ptr<LossFunction> m_loss_fn;
    AdvancedTrainingConfig m_config;
    
    // Advanced components
    std::unique_ptr<MixedPrecisionTrainer> m_mp_trainer;
    std::unique_ptr<RegularizationManager> m_regularization;
    std::unique_ptr<LearningRateScheduler> m_lr_scheduler;
    std::vector<std::unique_ptr<TrainingCallback>> m_callbacks;
    
    // Training state
    TrainingMetrics m_current_metrics;
    std::vector<TrainingMetrics> m_training_history;
    float m_best_validation_loss;
    int m_epochs_without_improvement;
    bool m_should_stop;
    
    // Internal methods
    void setup_mixed_precision();
    void setup_regularization();
    void setup_lr_scheduling();
    void setup_callbacks();
    
    TrainingMetrics process_epoch(data::DataLoader& train_loader,
                                 data::DataLoader& val_loader,
                                 int epoch);
    
    TrainingMetrics process_batch(const Tensor& inputs, const Tensor& targets,
                                 bool training = true);
    
    void update_learning_rate(float validation_metric);
    void save_checkpoint(int epoch, bool is_best = false);
    void log_metrics(const TrainingMetrics& metrics, bool is_validation = false);
    
public:
    AdvancedTrainer(std::shared_ptr<Model> model,
                   std::shared_ptr<Optimizer> optimizer,
                   std::shared_ptr<LossFunction> loss_fn,
                   const AdvancedTrainingConfig& config = {})
        : m_model(model), m_optimizer(optimizer), m_loss_fn(loss_fn),
          m_config(config), m_best_validation_loss(std::numeric_limits<float>::max()),
          m_epochs_without_improvement(0), m_should_stop(false) {
        setup_mixed_precision();
        setup_regularization();
        setup_lr_scheduling();
        setup_callbacks();
    }
    
    /**
     * @brief Train model with advanced features
     */
    std::vector<TrainingMetrics> fit(data::DataLoader& train_loader,
                                   data::DataLoader& val_loader,
                                   int epochs);
    
    /**
     * @brief Evaluate model
     */
    TrainingMetrics evaluate(data::DataLoader& data_loader);
    
    /**
     * @brief Get training history
     */
    const std::vector<TrainingMetrics>& get_training_history() const {
        return m_training_history;
    }
    
    /**
     * @brief Get current configuration
     */
    const AdvancedTrainingConfig& get_config() const { return m_config; }
    
    /**
     * @brief Update configuration
     */
    void update_config(const AdvancedTrainingConfig& config);
    
    /**
     * @brief Add custom callback
     */
    void add_callback(std::unique_ptr<TrainingCallback> callback);
    
    /**
     * @brief Get training statistics
     */
    struct TrainingStatistics {
        float total_training_time = 0.0f;
        float average_epoch_time = 0.0f;
        float best_validation_loss = 0.0f;
        float best_validation_accuracy = 0.0f;
        int best_epoch = 0;
        int total_epochs_trained = 0;
        bool converged = false;
        
        // Mixed precision stats
        float memory_savings_ratio = 0.0f;
        float training_speedup = 0.0f;
        
        // Regularization stats
        float final_l1_loss = 0.0f;
        float final_l2_loss = 0.0f;
        float final_weight_decay = 0.0f;
    };
    TrainingStatistics get_statistics() const;
};

/**
 * @brief Factory function for creating advanced trainer
 */
std::unique_ptr<AdvancedTrainer> create_advanced_trainer(
    std::shared_ptr<Model> model,
    const std::string& optimizer_name = "adam",
    float learning_rate = 0.001f,
    const std::string& loss_name = "crossentropy",
    const AdvancedTrainingConfig& config = {}
);

/**
 * @brief Hyperparameter tuning utilities
 */
namespace hyperparameter_tuning {
    
    struct HyperparameterRange {
        float min_value;
        float max_value;
        bool log_scale = false;
    };
    
    struct HyperparameterConfig {
        std::unordered_map<std::string, HyperparameterRange> ranges;
        int n_trials = 100;
        int max_epochs_per_trial = 10;
        std::string optimization_metric = "val_loss";
        bool maximize_metric = false;
    };
    
    /**
     * @brief Simple random search for hyperparameters
     */
    struct HyperparameterResult {
        std::unordered_map<std::string, float> best_params;
        float best_score;
        std::vector<std::pair<std::unordered_map<std::string, float>, float>> all_results;
    };
    
    HyperparameterResult random_search(
        std::function<float(const std::unordered_map<std::string, float>&)> objective_fn,
        const HyperparameterConfig& config
    );
}

} // namespace training
} // namespace dlvk
