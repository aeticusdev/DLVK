#include "dlvk/model/model.h"
#include "dlvk/layers/dense_layer.h"
#include "dlvk/layers/conv2d_layer.h"
#include "dlvk/layers/pooling_layers.h"
#include "dlvk/layers/batch_norm_layers.h"
#include "dlvk/layers/dropout_layer.h"
#include "dlvk/layers/activation.h"
#include "dlvk/layers/layer_adapters.h"
#include "dlvk/core/vulkan_device.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>

namespace dlvk {

// Sequential Model Implementation
Sequential::Sequential(std::shared_ptr<VulkanDevice> device)
    : m_device(device), m_is_training(false) {}

void Sequential::add(std::unique_ptr<ModernLayer> layer) {
    m_layers.push_back(std::move(layer));
}

void Sequential::add_dense(size_t input_size, size_t output_size, bool use_bias) {
    auto adapter = std::make_unique<DenseLayerAdapter>(*m_device, input_size, output_size, use_bias);
    add(std::move(adapter));
}

void Sequential::add_conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, 
                           size_t stride, size_t padding) {
    auto adapter = std::make_unique<Conv2DLayerAdapter>(*m_device, in_channels, out_channels, 
                                                        kernel_size, stride, padding);
    add(std::move(adapter));
}

void Sequential::add_maxpool2d(size_t pool_size, size_t stride) {
    if (stride == 0) stride = pool_size;
    auto adapter = std::make_unique<MaxPool2DLayerAdapter>(*m_device, pool_size, stride);
    add(std::move(adapter));
}

void Sequential::add_avgpool2d(size_t pool_size, size_t stride) {
    if (stride == 0) stride = pool_size;
    auto adapter = std::make_unique<AvgPool2DLayerAdapter>(*m_device, pool_size, stride);
    add(std::move(adapter));
}

void Sequential::add_batchnorm1d(size_t num_features) {
    auto adapter = std::make_unique<BatchNorm1DLayerAdapter>(*m_device, num_features);
    add(std::move(adapter));
}

void Sequential::add_batchnorm2d(size_t num_features) {
    auto adapter = std::make_unique<BatchNorm2DLayerAdapter>(*m_device, num_features);
    add(std::move(adapter));
}

void Sequential::add_dropout(float dropout_rate) {
    auto adapter = std::make_unique<DropoutLayerAdapter>(*m_device, dropout_rate);
    add(std::move(adapter));
}

void Sequential::add_relu() {
    auto layer = std::make_unique<ActivationLayer>(m_device, ActivationType::ReLU);
    add(std::move(layer));
}

void Sequential::add_sigmoid() {
    auto layer = std::make_unique<ActivationLayer>(m_device, ActivationType::Sigmoid);
    add(std::move(layer));
}

void Sequential::add_tanh() {
    auto layer = std::make_unique<ActivationLayer>(m_device, ActivationType::Tanh);
    add(std::move(layer));
}

void Sequential::add_softmax() {
    auto layer = std::make_unique<ActivationLayer>(m_device, ActivationType::Softmax);
    add(std::move(layer));
}

Tensor Sequential::forward(const Tensor& input) {
    if (m_layers.empty()) {
        throw std::runtime_error("No layers in the sequential model");
    }
    
    // Clear previous layer outputs
    m_layer_outputs.clear();
    m_layer_outputs.reserve(m_layers.size());
    
    Tensor current_input = input;
    
    for (size_t i = 0; i < m_layers.size(); ++i) {
        m_layers[i]->set_training(m_is_training);
        current_input = m_layers[i]->forward(current_input);
        m_layer_outputs.push_back(current_input);
    }
    
    return current_input;
}

void Sequential::backward(const Tensor& grad_output) {
    if (m_layers.empty()) {
        throw std::runtime_error("No layers in the sequential model");
    }
    
    Tensor current_grad = grad_output;
    
    // Backward pass through layers in reverse order
    for (int i = m_layers.size() - 1; i >= 0; --i) {
        current_grad = m_layers[i]->backward(current_grad);
    }
}

void Sequential::update_parameters(Optimizer& optimizer) {
    for (auto& layer : m_layers) {
        layer->update_parameters(optimizer);
    }
}

void Sequential::set_training(bool training) {
    m_is_training = training;
    for (auto& layer : m_layers) {
        layer->set_training(training);
    }
}

std::string Sequential::summary() const {
    std::stringstream ss;
    ss << "Sequential Model Summary\n";
    ss << "========================\n";
    
    size_t total_params = 0;
    size_t trainable_params = 0;
    
    for (size_t i = 0; i < m_layers.size(); ++i) {
        auto layer_info = m_layers[i]->get_layer_info();
        ss << "Layer " << std::setw(2) << i << ": " << std::setw(12) << layer_info.type 
           << " | Output: " << layer_info.output_shape_str 
           << " | Params: " << layer_info.parameter_count << "\n";
        
        total_params += layer_info.parameter_count;
        if (layer_info.trainable) {
            trainable_params += layer_info.parameter_count;
        }
    }
    
    ss << "========================\n";
    ss << "Total params: " << total_params << "\n";
    ss << "Trainable params: " << trainable_params << "\n";
    ss << "Non-trainable params: " << (total_params - trainable_params) << "\n";
    
    return ss.str();
}

size_t Sequential::parameter_count() const {
    size_t count = 0;
    for (const auto& layer : m_layers) {
        count += layer->get_layer_info().parameter_count;
    }
    return count;
}

void Sequential::save_weights(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
    
    // Write number of layers
    size_t num_layers = m_layers.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    // Save each layer's weights
    for (const auto& layer : m_layers) {
        layer->save_weights(file);
    }
    
    file.close();
}

void Sequential::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filepath);
    }
    
    // Read number of layers
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    if (num_layers != m_layers.size()) {
        throw std::runtime_error("Model architecture mismatch: expected " + 
                                std::to_string(m_layers.size()) + " layers, got " + 
                                std::to_string(num_layers));
    }
    
    // Load each layer's weights
    for (auto& layer : m_layers) {
        layer->load_weights(file);
    }
    
    file.close();
}

// ModelTrainer Implementation
ModelTrainer::ModelTrainer(Model* model) : m_model(model) {}

void ModelTrainer::compile(std::unique_ptr<Optimizer> optimizer, 
                          std::unique_ptr<LossFunction> loss_function) {
    m_optimizer = std::move(optimizer);
    m_loss_function = std::move(loss_function);
}

void ModelTrainer::add_callback(std::unique_ptr<TrainingCallback> callback) {
    m_callbacks.push_back(std::move(callback));
}

void ModelTrainer::fit(const Tensor& x_train, const Tensor& y_train, 
                       size_t epochs, size_t batch_size, float validation_split, bool verbose) {
    if (!m_optimizer || !m_loss_function) {
        throw std::runtime_error("Optimizer and loss function must be set before training");
    }
    
    // Split data if validation split is provided
    std::vector<size_t> dummy_shape = {1, 1};
    Tensor x_val(dummy_shape, x_train.dtype(), x_train.device());
    Tensor y_val(dummy_shape, y_train.dtype(), y_train.device());
    Tensor x_actual_train = x_train;
    Tensor y_actual_train = y_train;
    
    if (validation_split > 0.0f && validation_split < 1.0f) {
        auto [x_train_split, y_train_split, x_val_split, y_val_split] = 
            split_data(x_train, y_train, validation_split);
        x_actual_train = std::move(x_train_split);
        y_actual_train = std::move(y_train_split);
        x_val = std::move(x_val_split);
        y_val = std::move(y_val_split);
    }
    
    // Create batches
    auto batches = create_batches(x_actual_train, y_actual_train, batch_size);
    
    // Training loop
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        size_t num_batches = batches.size();
        
        // Call epoch begin callbacks
        for (auto& callback : m_callbacks) {
            callback->on_epoch_begin(epoch);
        }
        
        m_model->set_training(true);
        
        // Process each batch
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            const auto& [x_batch, y_batch] = batches[batch_idx];
            
            // Forward pass
            Tensor predictions = m_model->forward(x_batch);
            
            // Calculate loss
            auto pred_ptr = std::make_shared<Tensor>(predictions);
            auto target_ptr = std::make_shared<Tensor>(y_batch);
            auto loss_result = m_loss_function->forward(pred_ptr, target_ptr);
            float batch_loss = 0.0f; // Extract scalar from loss tensor
            if (loss_result && loss_result->size() == 1) {
                loss_result->download_data(&batch_loss);
            }
            epoch_loss += batch_loss;
            
            // Backward pass through loss
            auto grad_result = m_loss_function->backward(pred_ptr, target_ptr);
            Tensor grad_output = grad_result ? *grad_result : Tensor({1}, DataType::FLOAT32, x_batch.device());
            
            // Backward pass through model
            m_model->backward(grad_output);
            
            // Update parameters
            m_model->update_parameters(*m_optimizer);
        }
        
        epoch_loss /= num_batches;
        
        // Validation if available
        float val_loss = 0.0f;
        float val_accuracy = 0.0f;
        if (validation_split > 0.0f) {
            TrainingMetrics val_metrics = evaluate(x_val, y_val, batch_size);
            val_loss = val_metrics.loss;
            val_accuracy = val_metrics.accuracy;
        }
        
        // Call epoch end callbacks
        TrainingMetrics epoch_metrics;
        epoch_metrics.loss = epoch_loss;
        epoch_metrics.accuracy = val_accuracy;
        for (auto& callback : m_callbacks) {
            callback->on_epoch_end(epoch, epoch_metrics);
        }
        
        // Verbose output
        if (verbose) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                     << " - loss: " << std::fixed << std::setprecision(4) << epoch_loss;
            if (validation_split > 0.0f) {
                std::cout << " - val_loss: " << val_loss 
                         << " - val_accuracy: " << val_accuracy;
            }
            std::cout << std::endl;
        }
    }
}

TrainingMetrics ModelTrainer::evaluate(const Tensor& x_test, const Tensor& y_test,
                                       size_t batch_size) {
    if (!m_loss_function) {
        throw std::runtime_error("Loss function must be set before evaluation");
    }
    
    m_model->set_training(false);
    
    auto batches = create_batches(x_test, y_test, batch_size);
    
    float total_loss = 0.0f;
    float total_accuracy = 0.0f;
    size_t num_samples = 0;
    
    for (const auto& [x_batch, y_batch] : batches) {
        // Forward pass
        Tensor predictions = m_model->forward(x_batch);
        
        // Calculate loss
        auto pred_ptr = std::make_shared<Tensor>(predictions);
        auto target_ptr = std::make_shared<Tensor>(y_batch);
        auto loss_result = m_loss_function->forward(pred_ptr, target_ptr);
        float batch_loss = 0.0f;
        if (loss_result && loss_result->size() == 1) {
            loss_result->download_data(&batch_loss);
        }
        total_loss += batch_loss * x_batch.shape()[0]; // Weight by batch size
        
        // Calculate accuracy
        float batch_accuracy = calculate_accuracy(predictions, y_batch);
        total_accuracy += batch_accuracy * x_batch.shape()[0];
        
        num_samples += x_batch.shape()[0];
    }
    
    TrainingMetrics metrics;
    metrics.loss = total_loss / num_samples;
    metrics.accuracy = total_accuracy / num_samples;
    return metrics;
}

Tensor ModelTrainer::predict(const Tensor& x, size_t batch_size) {
    m_model->set_training(false);
    
    // For simplicity, if batch size is larger than input, just process all at once
    if (batch_size >= x.shape()[0]) {
        return m_model->forward(x);
    }
    
    // Otherwise, process in batches and concatenate results
    std::vector<size_t> result_shape = x.shape();
    // This is a simplified version - in practice we would need to adjust output shape
    // based on the model's output dimensions
    
    // For now, just return the forward pass of the entire input
    // TODO: Implement proper batched prediction with result concatenation
    return m_model->forward(x);
}

float ModelTrainer::calculate_accuracy(const Tensor& predictions, const Tensor& targets) {
    // Simple accuracy calculation for classification
    // This assumes predictions and targets are in the same format
    
    if (predictions.shape() != targets.shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape for accuracy calculation");
    }
    
    size_t num_samples = predictions.shape()[0];
    size_t num_classes = predictions.shape().size() > 1 ? predictions.shape()[1] : 1;
    
    // Download data to CPU for accuracy calculation
    std::vector<float> pred_data(predictions.size());
    std::vector<float> target_data(targets.size());
    
    predictions.download_data(pred_data.data());
    targets.download_data(target_data.data());
    
    size_t correct_predictions = 0;
    
    if (num_classes == 1) {
        // Binary classification
        for (size_t i = 0; i < num_samples; ++i) {
            bool pred_class = pred_data[i] > 0.5f;
            bool target_class = target_data[i] > 0.5f;
            if (pred_class == target_class) {
                correct_predictions++;
            }
        }
    } else {
        // Multi-class classification
        for (size_t i = 0; i < num_samples; ++i) {
            // Find predicted class (argmax)
            size_t pred_class = 0;
            float max_pred = pred_data[i * num_classes];
            for (size_t j = 1; j < num_classes; ++j) {
                if (pred_data[i * num_classes + j] > max_pred) {
                    max_pred = pred_data[i * num_classes + j];
                    pred_class = j;
                }
            }
            
            // Find target class (argmax)
            size_t target_class = 0;
            float max_target = target_data[i * num_classes];
            for (size_t j = 1; j < num_classes; ++j) {
                if (target_data[i * num_classes + j] > max_target) {
                    max_target = target_data[i * num_classes + j];
                    target_class = j;
                }
            }
            
            if (pred_class == target_class) {
                correct_predictions++;
            }
        }
    }
    
    return static_cast<float>(correct_predictions) / static_cast<float>(num_samples);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> ModelTrainer::split_data(
    const Tensor& x, const Tensor& y, float validation_split) {
    
    size_t total_samples = x.shape()[0];
    size_t val_samples = static_cast<size_t>(total_samples * validation_split);
    size_t train_samples = total_samples - val_samples;
    
    // Create indices for shuffling
    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Split based on shuffled indices
    // For simplicity, just take first/last portions
    // TODO: Implement proper tensor slicing and shuffling
    
    // Create shape vectors for training and validation data
    std::vector<size_t> x_train_shape = x.shape();
    x_train_shape[0] = train_samples;
    
    std::vector<size_t> x_val_shape = x.shape();
    x_val_shape[0] = val_samples;
    
    std::vector<size_t> y_train_shape = y.shape();
    y_train_shape[0] = train_samples;
    
    std::vector<size_t> y_val_shape = y.shape();
    y_val_shape[0] = val_samples;
    
    // For now, just return the original tensors as train/val split
    // TODO: Implement proper data copying with shuffling using Vulkan operations
    
    // Create simple sliced views (placeholder implementation)
    // In a full implementation, we would use Vulkan compute shaders to slice tensors
    Tensor x_train(x_train_shape, x.dtype(), x.device());
    Tensor y_train(y_train_shape, y.dtype(), y.device());
    Tensor x_val(x_val_shape, x.dtype(), x.device());
    Tensor y_val(y_val_shape, y.dtype(), y.device());
    
    // For simplified implementation, copy entire datasets for now
    // TODO: Implement proper slicing and data copying
    std::vector<float> temp_data(x.size());
    x.download_data(temp_data.data());
    x_train.upload_data(temp_data.data());
    x_val.upload_data(temp_data.data());
    
    std::vector<float> temp_y_data(y.size()); 
    y.download_data(temp_y_data.data());
    y_train.upload_data(temp_y_data.data());
    y_val.upload_data(temp_y_data.data());
    
    return std::make_tuple(std::move(x_train), std::move(y_train), 
                          std::move(x_val), std::move(y_val));
}

std::vector<std::pair<Tensor, Tensor>> ModelTrainer::create_batches(
    const Tensor& x, const Tensor& y, size_t batch_size) {
    
    size_t total_samples = x.shape()[0];
    size_t num_batches = (total_samples + batch_size - 1) / batch_size;
    
    std::vector<std::pair<Tensor, Tensor>> batches;
    batches.reserve(num_batches);
    
    for (size_t i = 0; i < num_batches; ++i) {
        size_t start_idx = i * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, total_samples);
        size_t actual_batch_size = end_idx - start_idx;
        
        // Create batch shapes
        std::vector<size_t> x_batch_shape = x.shape();
        x_batch_shape[0] = actual_batch_size;
        
        std::vector<size_t> y_batch_shape = y.shape();
        y_batch_shape[0] = actual_batch_size;
        
        Tensor x_batch(x_batch_shape, x.dtype(), x.device());
        Tensor y_batch(y_batch_shape, y.dtype(), y.device());
        
        // For simplified implementation, copy entire data for now
        // TODO: Implement proper batching with Vulkan compute shaders
        std::vector<float> temp_x_data(x.size());
        std::vector<float> temp_y_data(y.size());
        x.download_data(temp_x_data.data());
        y.download_data(temp_y_data.data());
        x_batch.upload_data(temp_x_data.data());
        y_batch.upload_data(temp_y_data.data());
        
        batches.emplace_back(std::move(x_batch), std::move(y_batch));
    }
    
    return batches;
}

} // namespace dlvk
