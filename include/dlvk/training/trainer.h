#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <chrono>
#include "dlvk/tensor/tensor.h"
#include "dlvk/model/model.h"
#include "dlvk/optimizers/optimizers.h"
#include "dlvk/loss/loss_functions.h"
#include "dlvk/data/dataloader.h"

namespace dlvk {
namespace training {

/**
 * @brief Training metrics for monitoring progress
 */
struct TrainingMetrics {
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;
    float val_loss = 0.0f;
    float val_accuracy = 0.0f;
    int epoch = 0;
    int batch = 0;
    std::chrono::milliseconds epoch_time{0};
    

    float training_loss = 0.0f;      // alias for train_loss
    float training_accuracy = 0.0f;  // alias for train_accuracy  
    float validation_loss = 0.0f;    // alias for val_loss
    float validation_accuracy = 0.0f; // alias for val_accuracy
    float learning_rate = 0.0f;
    float epoch_time_ms = 0.0f;      // epoch time in milliseconds
};

/**
 * @brief Callback interface for training events
 */
class TrainingCallback {
public:
    virtual ~TrainingCallback() = default;
    
    virtual void on_epoch_begin(int epoch) {}
    virtual void on_epoch_end(int epoch, const TrainingMetrics& metrics) {}
    virtual void on_batch_begin(int batch) {}
    virtual void on_batch_end(int batch, const TrainingMetrics& metrics) {}
    virtual void on_training_begin() {}
    virtual void on_training_end() {}
};

/**
 * @brief Progress callback for training visualization
 */
class ProgressCallback : public TrainingCallback {
private:
    int m_total_epochs;
    int m_print_every;
    
public:
    ProgressCallback(int total_epochs, int print_every = 1) 
        : m_total_epochs(total_epochs), m_print_every(print_every) {}
    
    void on_training_begin() override;
    void on_epoch_end(int epoch, const TrainingMetrics& metrics) override;
    void on_training_end() override;
};

/**
 * @brief Early stopping callback
 */
class EarlyStoppingCallback : public TrainingCallback {
private:
    float m_best_loss;
    int m_patience;
    int m_wait;
    bool m_should_stop;
    
public:
    EarlyStoppingCallback(int patience = 10) 
        : m_patience(patience), m_wait(0), m_should_stop(false), 
          m_best_loss(std::numeric_limits<float>::max()) {}
    
    void on_epoch_end(int epoch, const TrainingMetrics& metrics) override;
    bool should_stop() const { return m_should_stop; }
};

/**
 * @brief Complete training loop implementation
 */
class Trainer {
private:
    std::shared_ptr<Model> m_model;
    std::shared_ptr<Optimizer> m_optimizer;
    std::shared_ptr<LossFunction> m_loss_fn;
    std::vector<std::unique_ptr<TrainingCallback>> m_callbacks;
    

    TrainingMetrics m_current_metrics;
    bool m_should_stop = false;
    

    float compute_accuracy(const Tensor& predictions, const Tensor& targets);
    

    TrainingMetrics process_batch(const Tensor& inputs, const Tensor& targets, bool training = true);
    
public:
    Trainer(std::shared_ptr<Model> model, 
            std::shared_ptr<Optimizer> optimizer,
            std::shared_ptr<LossFunction> loss_fn)
        : m_model(model), m_optimizer(optimizer), m_loss_fn(loss_fn) {}
    

    void add_callback(std::unique_ptr<TrainingCallback> callback);
    

    void fit(data::DataLoader& train_loader,
             data::DataLoader& val_loader,
             int epochs,
             bool verbose = true);
    

    TrainingMetrics evaluate(data::DataLoader& data_loader, bool verbose = true);
    

    const TrainingMetrics& get_metrics() const { return m_current_metrics; }
};

/**
 * @brief Factory function for common training setup
 */
std::unique_ptr<Trainer> create_trainer(
    std::shared_ptr<Model> model,
    const std::string& optimizer_name = "adam",
    float learning_rate = 0.001f,
    const std::string& loss_name = "crossentropy"
);

} // namespace training
} // namespace dlvk
