#include "dlvk/loss/loss_functions.h"
#include <iostream>
#include <vector>
#include <cmath>

namespace dlvk {

// Mean Squared Error Implementation
std::shared_ptr<Tensor> MeanSquaredError::forward(const std::shared_ptr<Tensor>& predictions, 
                                                 const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    // MSE = mean((predictions - targets)^2)
    
    // Create result tensor for scalar loss
    auto device = predictions->device();
    auto loss = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, device);
    
    // For now, implement on CPU
    std::vector<float> pred_data(predictions->size());
    std::vector<float> target_data(targets->size());
    
    predictions->download_data(pred_data.data());
    targets->download_data(target_data.data());
    
    float mse = 0.0f;
    for (size_t i = 0; i < pred_data.size(); ++i) {
        float diff = pred_data[i] - target_data[i];
        mse += diff * diff;
    }
    mse /= static_cast<float>(pred_data.size());
    
    loss->upload_data(&mse);
    return loss;
}

std::shared_ptr<Tensor> MeanSquaredError::backward(const std::shared_ptr<Tensor>& predictions,
                                                  const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    // Gradient: d/dpred MSE = 2 * (predictions - targets) / n
    auto gradient = std::make_shared<Tensor>(predictions->shape(), DataType::FLOAT32, predictions->device());
    
    // For now, implement on CPU
    std::vector<float> pred_data(predictions->size());
    std::vector<float> target_data(targets->size());
    std::vector<float> grad_data(predictions->size());
    
    predictions->download_data(pred_data.data());
    targets->download_data(target_data.data());
    
    float scale = 2.0f / static_cast<float>(pred_data.size());
    for (size_t i = 0; i < pred_data.size(); ++i) {
        grad_data[i] = scale * (pred_data[i] - target_data[i]);
    }
    
    gradient->upload_data(grad_data.data());
    return gradient;
}

// Cross Entropy Loss Implementation
std::shared_ptr<Tensor> CrossEntropyLoss::forward(const std::shared_ptr<Tensor>& predictions, 
                                                  const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    // For now, implement cross-entropy on CPU
    // loss = -sum(targets * log(predictions + epsilon))
    std::vector<float> pred_data(predictions->size());
    std::vector<float> target_data(targets->size());
    
    predictions->download_data(pred_data.data());
    targets->download_data(target_data.data());
    
    float total_loss = 0.0f;
    const float epsilon = 1e-8f; // For numerical stability
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        total_loss -= target_data[i] * std::log(pred_data[i] + epsilon);
    }
    
    // Average loss
    total_loss /= static_cast<float>(predictions->shape()[0]); // Batch size
    
    auto result = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, predictions->device());
    result->upload_data(&total_loss);
    
    return result;
}

std::shared_ptr<Tensor> CrossEntropyLoss::backward(const std::shared_ptr<Tensor>& predictions,
                                                   const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    // Cross-entropy gradient: grad = (predictions - targets) / batch_size
    auto grad = predictions->subtract(*targets);
    float batch_size = static_cast<float>(predictions->shape()[0]);
    auto grad_normalized = grad->multiply_scalar(1.0f / batch_size);
    
    return grad_normalized;
}

// Binary Cross-Entropy Loss Implementation
std::shared_ptr<Tensor> BinaryCrossEntropyLoss::forward(const std::shared_ptr<Tensor>& predictions,
                                                        const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    // Download data
    std::vector<float> pred_data(predictions->size());
    std::vector<float> target_data(targets->size());
    predictions->download_data(pred_data.data());
    targets->download_data(target_data.data());
    
    // Compute binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        // Clamp predictions to avoid log(0)
        float p = std::max(epsilon_, std::min(1.0f - epsilon_, pred_data[i]));
        float y = target_data[i];
        
        total_loss -= y * std::log(p) + (1.0f - y) * std::log(1.0f - p);
    }
    
    // Average loss
    total_loss /= static_cast<float>(predictions->shape()[0]); // Batch size
    
    auto result = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, predictions->device());
    result->upload_data(&total_loss);
    
    return result;
}

std::shared_ptr<Tensor> BinaryCrossEntropyLoss::backward(const std::shared_ptr<Tensor>& predictions,
                                                         const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    // Download data
    std::vector<float> pred_data(predictions->size());
    std::vector<float> target_data(targets->size());
    predictions->download_data(pred_data.data());
    targets->download_data(target_data.data());
    
    // Compute gradient: (p - y) / (p * (1 - p)) / batch_size
    std::vector<float> grad_data(predictions->size());
    float batch_size = static_cast<float>(predictions->shape()[0]);
    
    for (size_t i = 0; i < pred_data.size(); ++i) {
        // Clamp predictions for numerical stability
        float p = std::max(epsilon_, std::min(1.0f - epsilon_, pred_data[i]));
        float y = target_data[i];
        
        grad_data[i] = (p - y) / (p * (1.0f - p)) / batch_size;
    }
    
    auto grad = std::make_shared<Tensor>(predictions->shape(), DataType::FLOAT32, predictions->device());
    grad->upload_data(grad_data.data());
    
    return grad;
}

} // namespace dlvk
