#include "dlvk/model/callbacks.h"
#include <iostream>
#include <iomanip>
#include <fstream>

namespace dlvk {

// ProgressCallback Implementation
ProgressCallback::ProgressCallback(bool verbose) 
    : m_total_epochs(0), m_total_batches_per_epoch(0), m_verbose(verbose) {}

void ProgressCallback::on_train_begin() {
    if (m_verbose) {
        std::cout << "Starting training...\n";
    }
}

void ProgressCallback::on_epoch_begin(size_t epoch) {
    m_epoch_start = std::chrono::steady_clock::now();
    if (m_verbose && m_total_batches_per_epoch > 0) {
        std::cout << "Epoch " << (epoch + 1) << "/" << m_total_epochs << ": ";
        std::cout.flush();
    }
}

void ProgressCallback::on_epoch_end([[maybe_unused]] size_t epoch, const TrainingMetrics& metrics) {
    auto epoch_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - m_epoch_start);
    
    if (m_verbose) {
        std::cout << " - " << duration.count() << "ms/epoch";
        std::cout << " - loss: " << std::fixed << std::setprecision(4) << metrics.loss;
        std::cout << " - accuracy: " << std::setprecision(4) << metrics.accuracy;
        
        if (metrics.validation_loss > 0) {
            std::cout << " - val_loss: " << std::setprecision(4) << metrics.validation_loss;
            std::cout << " - val_accuracy: " << std::setprecision(4) << metrics.validation_accuracy;
        }
        std::cout << std::endl;
    }
}

void ProgressCallback::on_batch_end(size_t batch, [[maybe_unused]] const TrainingMetrics& metrics) {
    if (m_verbose && m_total_batches_per_epoch > 10) {
        // Show progress for long epochs
        if (batch % (m_total_batches_per_epoch / 10) == 0) {
            float progress = static_cast<float>(batch + 1) / m_total_batches_per_epoch;
            int bar_width = 30;
            int pos = static_cast<int>(bar_width * progress);
            
            std::cout << "\r[";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "%";
            std::cout.flush();
        }
    }
}

// ModelCheckpoint Implementation
ModelCheckpoint::ModelCheckpoint(const std::string& filepath, Model* model,
                               const std::string& monitor,
                               bool save_best_only,
                               bool save_weights_only)
    : m_filepath(filepath), m_monitor(monitor), m_save_best_only(save_best_only),
      m_save_weights_only(save_weights_only), m_model(model) {
    
    // Initialize best score based on monitor type
    m_higher_is_better = (monitor == "accuracy" || monitor == "val_accuracy");
    m_best_score = m_higher_is_better ? -std::numeric_limits<float>::infinity() 
                                      : std::numeric_limits<float>::infinity();
}

void ModelCheckpoint::on_epoch_end(size_t epoch, const TrainingMetrics& metrics) {
    float current_value = get_monitored_value(metrics);
    
    bool should_save = !m_save_best_only || is_improvement(current_value);
    
    if (should_save) {
        if (m_save_best_only) {
            m_best_score = current_value;
        }
        
        try {
            if (m_save_weights_only) {
                m_model->save_weights(m_filepath);
            }
            // TODO: Implement full model saving
            
            std::cout << "\nEpoch " << (epoch + 1) << ": " << m_monitor 
                     << " improved to " << std::fixed << std::setprecision(4) 
                     << current_value << ", saving model to " << m_filepath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error saving model: " << e.what() << std::endl;
        }
    }
}

float ModelCheckpoint::get_monitored_value(const TrainingMetrics& metrics) const {
    if (m_monitor == "loss") return metrics.loss;
    if (m_monitor == "accuracy") return metrics.accuracy;
    if (m_monitor == "val_loss") return metrics.validation_loss;
    if (m_monitor == "val_accuracy") return metrics.validation_accuracy;
    
    throw std::runtime_error("Unknown monitor metric: " + m_monitor);
}

bool ModelCheckpoint::is_improvement(float current_value) const {
    if (m_higher_is_better) {
        return current_value > m_best_score;
    } else {
        return current_value < m_best_score;
    }
}

// EarlyStopping Implementation
EarlyStopping::EarlyStopping(const std::string& monitor, size_t patience, float min_delta)
    : m_monitor(monitor), m_patience(patience), m_min_delta(min_delta),
      m_wait_count(0), m_should_stop(false) {
    
    m_higher_is_better = (monitor == "accuracy" || monitor == "val_accuracy");
    m_best_score = m_higher_is_better ? -std::numeric_limits<float>::infinity() 
                                      : std::numeric_limits<float>::infinity();
}

void EarlyStopping::on_train_begin() {
    m_wait_count = 0;
    m_should_stop = false;
    m_best_score = m_higher_is_better ? -std::numeric_limits<float>::infinity() 
                                      : std::numeric_limits<float>::infinity();
}

void EarlyStopping::on_epoch_end(size_t epoch, const TrainingMetrics& metrics) {
    float current_value = get_monitored_value(metrics);
    
    if (is_improvement(current_value)) {
        m_best_score = current_value;
        m_wait_count = 0;
    } else {
        m_wait_count++;
        
        if (m_wait_count >= m_patience) {
            m_should_stop = true;
            std::cout << "\nEarly stopping triggered after " << (epoch + 1) 
                     << " epochs. Best " << m_monitor << ": " 
                     << std::fixed << std::setprecision(4) << m_best_score << std::endl;
        }
    }
}

float EarlyStopping::get_monitored_value(const TrainingMetrics& metrics) const {
    if (m_monitor == "loss") return metrics.loss;
    if (m_monitor == "accuracy") return metrics.accuracy;
    if (m_monitor == "val_loss") return metrics.validation_loss;
    if (m_monitor == "val_accuracy") return metrics.validation_accuracy;
    
    throw std::runtime_error("Unknown monitor metric: " + m_monitor);
}

bool EarlyStopping::is_improvement(float current_value) const {
    if (m_higher_is_better) {
        return current_value > (m_best_score + m_min_delta);
    } else {
        return current_value < (m_best_score - m_min_delta);
    }
}

// ReduceLROnPlateau Implementation
ReduceLROnPlateau::ReduceLROnPlateau(Optimizer* optimizer,
                                   const std::string& monitor,
                                   float factor,
                                   size_t patience,
                                   float min_delta,
                                   float min_lr)
    : m_monitor(monitor), m_factor(factor), m_patience(patience), 
      m_min_delta(min_delta), m_min_lr(min_lr), m_wait_count(0),
      m_best_score(0.0f), m_higher_is_better(false), m_optimizer(optimizer) {
    
    m_higher_is_better = (monitor == "accuracy" || monitor == "val_accuracy");
    m_best_score = m_higher_is_better ? -std::numeric_limits<float>::infinity() 
                                      : std::numeric_limits<float>::infinity();
}

void ReduceLROnPlateau::on_train_begin() {
    m_wait_count = 0;
    m_best_score = m_higher_is_better ? -std::numeric_limits<float>::infinity() 
                                      : std::numeric_limits<float>::infinity();
}

void ReduceLROnPlateau::on_epoch_end(size_t epoch, const TrainingMetrics& metrics) {
    float current_value = get_monitored_value(metrics);
    
    if (is_improvement(current_value)) {
        m_best_score = current_value;
        m_wait_count = 0;
    } else {
        m_wait_count++;
        
        if (m_wait_count >= m_patience) {
            float current_lr = m_optimizer->get_learning_rate();
            float new_lr = current_lr * m_factor;
            
            if (new_lr >= m_min_lr) {
                m_optimizer->set_learning_rate(new_lr);
                std::cout << "\nEpoch " << (epoch + 1) << ": reducing learning rate from " 
                         << current_lr << " to " << new_lr << std::endl;
                m_wait_count = 0;
            }
        }
    }
}

float ReduceLROnPlateau::get_monitored_value(const TrainingMetrics& metrics) const {
    if (m_monitor == "loss") return metrics.loss;
    if (m_monitor == "accuracy") return metrics.accuracy;
    if (m_monitor == "val_loss") return metrics.validation_loss;
    if (m_monitor == "val_accuracy") return metrics.validation_accuracy;
    
    throw std::runtime_error("Unknown monitor metric: " + m_monitor);
}

bool ReduceLROnPlateau::is_improvement(float current_value) const {
    if (m_higher_is_better) {
        return current_value > (m_best_score + m_min_delta);
    } else {
        return current_value < (m_best_score - m_min_delta);
    }
}

// CSVLogger Implementation
CSVLogger::CSVLogger(const std::string& filename, bool append)
    : m_filename(filename), m_append(append) {}

CSVLogger::~CSVLogger() {
    if (m_file.is_open()) {
        m_file.close();
    }
}

void CSVLogger::on_train_begin() {
    m_file.open(m_filename, m_append ? std::ios::app : std::ios::out);
    
    if (!m_file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + m_filename);
    }
    
    if (!m_append || m_file.tellp() == 0) {
        // Write header
        m_file << "epoch,loss,accuracy,val_loss,val_accuracy\n";
    }
}

void CSVLogger::on_epoch_end(size_t epoch, const TrainingMetrics& metrics) {
    m_file << epoch << "," 
           << metrics.loss << "," 
           << metrics.accuracy << "," 
           << metrics.validation_loss << "," 
           << metrics.validation_accuracy << "\n";
    m_file.flush();
}

} // namespace dlvk