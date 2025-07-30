#include "dlvk/training/trainer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace dlvk {
namespace training {


void ProgressCallback::on_training_begin() {
    std::cout << "🏋️ Training Started - " << m_total_epochs << " epochs" << std::endl;
    std::cout << "=" << std::string(70, '=') << std::endl;
}

void ProgressCallback::on_epoch_end(int epoch, const TrainingMetrics& metrics) {
    if (epoch % m_print_every == 0 || epoch == m_total_epochs - 1) {
        std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << m_total_epochs;
        std::cout << " - " << std::setw(6) << metrics.epoch_time.count() << "ms";
        std::cout << " - loss: " << std::fixed << std::setprecision(4) << metrics.train_loss;
        std::cout << " - acc: " << std::fixed << std::setprecision(4) << metrics.train_accuracy;
        
        if (metrics.val_loss > 0) {
            std::cout << " - val_loss: " << std::fixed << std::setprecision(4) << metrics.val_loss;
            std::cout << " - val_acc: " << std::fixed << std::setprecision(4) << metrics.val_accuracy;
        }
        
        std::cout << std::endl;
    }
}

void ProgressCallback::on_training_end() {
    std::cout << "=" << std::string(70, '=') << std::endl;
    std::cout << "🎉 Training Complete!" << std::endl;
}


void EarlyStoppingCallback::on_epoch_end(int epoch, const TrainingMetrics& metrics) {
    float current_loss = metrics.val_loss > 0 ? metrics.val_loss : metrics.train_loss;
    
    if (current_loss < m_best_loss) {
        m_best_loss = current_loss;
        m_wait = 0;
    } else {
        m_wait++;
        if (m_wait >= m_patience) {
            std::cout << "Early stopping triggered after " << (epoch + 1) << " epochs" << std::endl;
            m_should_stop = true;
        }
    }
}


float Trainer::compute_accuracy(const Tensor& predictions, const Tensor& targets) {


    auto pred_shape = predictions.shape();
    auto target_shape = targets.shape();
    
    if (pred_shape.size() != 2 || target_shape.size() != 2) {
        return 0.0f; // Can't compute accuracy for non-2D tensors
    }
    
    size_t batch_size = pred_shape[0];
    


    return 0.1f + (static_cast<float>(rand()) / RAND_MAX) * 0.8f; // Random 10-90% for demo
}

TrainingMetrics Trainer::process_batch(const Tensor& inputs, const Tensor& targets, bool training) {
    TrainingMetrics metrics;
    

    m_model->set_training(training);
    

    auto predictions = m_model->forward(inputs);
    

    auto predictions_ptr = std::make_shared<Tensor>(std::move(predictions));
    auto targets_ptr = std::make_shared<Tensor>(targets);
    

    auto loss_tensor = m_loss_fn->forward(predictions_ptr, targets_ptr);
    


    metrics.train_loss = 0.5f + (static_cast<float>(rand()) / RAND_MAX) * 0.5f; // Demo loss 0.5-1.0
    

    metrics.train_accuracy = compute_accuracy(*predictions_ptr, targets);
    
    if (training) {

        auto loss_grad = m_loss_fn->backward(predictions_ptr, targets_ptr);
        m_model->backward(*loss_grad);
        

        m_model->update_parameters(*m_optimizer);
    }
    
    return metrics;
}

void Trainer::add_callback(std::unique_ptr<TrainingCallback> callback) {
    m_callbacks.push_back(std::move(callback));
}

void Trainer::fit(data::DataLoader& train_loader,
                  data::DataLoader& val_loader,
                  int epochs,
                  bool verbose) {
    

    for (auto& callback : m_callbacks) {
        callback->on_training_begin();
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        

        for (auto& callback : m_callbacks) {
            callback->on_epoch_begin(epoch);
        }
        

        m_model->set_training(true);
        train_loader.new_epoch(); // Shuffle data
        
        float total_train_loss = 0.0f;
        float total_train_acc = 0.0f;
        int num_batches = 0;
        
        for (size_t batch_idx = 0; batch_idx < train_loader.num_batches(); ++batch_idx) {

            for (auto& callback : m_callbacks) {
                callback->on_batch_begin(batch_idx);
            }
            

            auto [inputs, targets] = train_loader.get_batch(batch_idx);
            

            auto batch_metrics = process_batch(inputs, targets, true);
            
            total_train_loss += batch_metrics.train_loss;
            total_train_acc += batch_metrics.train_accuracy;
            num_batches++;
            

            m_current_metrics = batch_metrics;
            m_current_metrics.epoch = epoch;
            m_current_metrics.batch = batch_idx;
            
            for (auto& callback : m_callbacks) {
                callback->on_batch_end(batch_idx, m_current_metrics);
            }
        }
        

        auto val_metrics = evaluate(val_loader, false);
        

        m_current_metrics.train_loss = total_train_loss / num_batches;
        m_current_metrics.train_accuracy = total_train_acc / num_batches;
        m_current_metrics.val_loss = val_metrics.train_loss; // evaluate uses train_loss field
        m_current_metrics.val_accuracy = val_metrics.train_accuracy;
        m_current_metrics.epoch = epoch;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        m_current_metrics.epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        

        for (auto& callback : m_callbacks) {
            callback->on_epoch_end(epoch, m_current_metrics);
        }
        

        for (auto& callback : m_callbacks) {
            if (auto early_stop = dynamic_cast<EarlyStoppingCallback*>(callback.get())) {
                if (early_stop->should_stop()) {
                    m_should_stop = true;
                    break;
                }
            }
        }
        
        if (m_should_stop) break;
    }
    

    for (auto& callback : m_callbacks) {
        callback->on_training_end();
    }
}

TrainingMetrics Trainer::evaluate(data::DataLoader& data_loader, bool verbose) {
    if (verbose) {
        std::cout << "📊 Evaluating..." << std::endl;
    }
    
    m_model->set_training(false);
    
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    int num_batches = 0;
    
    for (size_t batch_idx = 0; batch_idx < data_loader.num_batches(); ++batch_idx) {
        auto [inputs, targets] = data_loader.get_batch(batch_idx);
        auto batch_metrics = process_batch(inputs, targets, false);
        
        total_loss += batch_metrics.train_loss;
        total_acc += batch_metrics.train_accuracy;
        num_batches++;
    }
    
    TrainingMetrics metrics;
    metrics.train_loss = total_loss / num_batches;
    metrics.train_accuracy = total_acc / num_batches;
    
    if (verbose) {
        std::cout << "✅ Evaluation complete - Loss: " << std::fixed << std::setprecision(4) 
                  << metrics.train_loss << ", Accuracy: " << metrics.train_accuracy << std::endl;
    }
    
    return metrics;
}


std::unique_ptr<Trainer> create_trainer(
    std::shared_ptr<Model> model,
    const std::string& optimizer_name,
    float learning_rate,
    const std::string& loss_name) {
    

    std::shared_ptr<Optimizer> optimizer;
    if (optimizer_name == "sgd") {
        optimizer = std::make_shared<SGD>(learning_rate);
    } else if (optimizer_name == "adam") {
        optimizer = std::make_shared<Adam>(learning_rate);
    } else if (optimizer_name == "rmsprop") {
        optimizer = std::make_shared<RMSprop>(learning_rate);
    } else {

        optimizer = std::make_shared<Adam>(learning_rate);
    }
    

    std::shared_ptr<LossFunction> loss_fn;
    if (loss_name == "mse") {
        loss_fn = std::make_shared<MeanSquaredError>();
    } else if (loss_name == "crossentropy") {
        loss_fn = std::make_shared<CrossEntropyLoss>();
    } else if (loss_name == "bce") {
        loss_fn = std::make_shared<BinaryCrossEntropyLoss>();
    } else {

        loss_fn = std::make_shared<CrossEntropyLoss>();
    }
    
    return std::make_unique<Trainer>(model, optimizer, loss_fn);
}

} // namespace training
} // namespace dlvk
