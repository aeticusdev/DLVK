#pragma once

#include "dlvk/model/model.h"
#include <string>
#include <vector>
#include <chrono>

namespace dlvk {

/**
 * @brief Progress bar callback for training visualization
 */
class ProgressCallback : public TrainingCallback {
private:
    size_t m_total_epochs;
    size_t m_total_batches_per_epoch;
    std::chrono::steady_clock::time_point m_epoch_start;
    bool m_verbose;
    
public:
    explicit ProgressCallback(bool verbose = true);
    
    void on_train_begin() override;
    void on_epoch_begin(size_t epoch) override;
    void on_epoch_end(size_t epoch, const TrainingMetrics& metrics) override;
    void on_batch_end(size_t batch, const TrainingMetrics& metrics) override;
    
    void set_total_epochs(size_t epochs) { m_total_epochs = epochs; }
    void set_batches_per_epoch(size_t batches) { m_total_batches_per_epoch = batches; }
};

/**
 * @brief Model checkpointing callback
 */
class ModelCheckpoint : public TrainingCallback {
private:
    std::string m_filepath;
    std::string m_monitor;
    bool m_save_best_only;
    bool m_save_weights_only;
    float m_best_score;
    bool m_higher_is_better;
    Model* m_model;
    
public:
    /**
     * @brief Constructor for model checkpoint callback
     * @param filepath Path to save the model weights
     * @param model Pointer to the model to save
     * @param monitor Metric to monitor for best model ('loss', 'accuracy', 'val_loss', 'val_accuracy')
     * @param save_best_only Whether to only save when the monitored metric improves
     * @param save_weights_only Whether to save only weights or the entire model
     */
    ModelCheckpoint(const std::string& filepath, Model* model,
                   const std::string& monitor = "val_loss",
                   bool save_best_only = true,
                   bool save_weights_only = true);
    
    void on_epoch_end(size_t epoch, const TrainingMetrics& metrics) override;
    
private:
    float get_monitored_value(const TrainingMetrics& metrics) const;
    bool is_improvement(float current_value) const;
};

/**
 * @brief Early stopping callback
 */
class EarlyStopping : public TrainingCallback {
private:
    std::string m_monitor;
    size_t m_patience;
    float m_min_delta;
    size_t m_wait_count;
    float m_best_score;
    bool m_higher_is_better;
    bool m_should_stop;
    
public:
    /**
     * @brief Constructor for early stopping callback
     * @param monitor Metric to monitor ('loss', 'accuracy', 'val_loss', 'val_accuracy')
     * @param patience Number of epochs with no improvement to wait before stopping
     * @param min_delta Minimum change in monitored metric to qualify as improvement
     */
    EarlyStopping(const std::string& monitor = "val_loss",
                 size_t patience = 10,
                 float min_delta = 0.0f);
    
    void on_train_begin() override;
    void on_epoch_end(size_t epoch, const TrainingMetrics& metrics) override;
    
    bool should_stop() const { return m_should_stop; }
    
private:
    float get_monitored_value(const TrainingMetrics& metrics) const;
    bool is_improvement(float current_value) const;
};

/**
 * @brief Learning rate reduction callback
 */
class ReduceLROnPlateau : public TrainingCallback {
private:
    std::string m_monitor;
    float m_factor;
    size_t m_patience;
    float m_min_delta;
    float m_min_lr;
    size_t m_wait_count;
    float m_best_score;
    bool m_higher_is_better;
    Optimizer* m_optimizer;
    
public:
    /**
     * @brief Constructor for learning rate reduction callback
     * @param optimizer Pointer to the optimizer to modify
     * @param monitor Metric to monitor ('loss', 'accuracy', 'val_loss', 'val_accuracy')
     * @param factor Factor by which to reduce the learning rate
     * @param patience Number of epochs with no improvement to wait before reducing LR
     * @param min_delta Minimum change in monitored metric to qualify as improvement
     * @param min_lr Lower bound on the learning rate
     */
    ReduceLROnPlateau(Optimizer* optimizer,
                     const std::string& monitor = "val_loss",
                     float factor = 0.1f,
                     size_t patience = 10,
                     float min_delta = 0.0f,
                     float min_lr = 0.0f);
    
    void on_train_begin() override;
    void on_epoch_end(size_t epoch, const TrainingMetrics& metrics) override;
    
private:
    float get_monitored_value(const TrainingMetrics& metrics) const;
    bool is_improvement(float current_value) const;
};

/**
 * @brief CSV logger callback for training metrics
 */
class CSVLogger : public TrainingCallback {
private:
    std::string m_filename;
    std::ofstream m_file;
    bool m_append;
    
public:
    /**
     * @brief Constructor for CSV logger callback
     * @param filename Path to the CSV file
     * @param append Whether to append to existing file or overwrite
     */
    CSVLogger(const std::string& filename, bool append = false);
    ~CSVLogger();
    
    void on_train_begin() override;
    void on_epoch_end(size_t epoch, const TrainingMetrics& metrics) override;
};

} // namespace dlvk
