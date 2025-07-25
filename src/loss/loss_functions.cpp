#include "dlvk/loss/loss_functions.h"
#include "dlvk/loss/loss_ops_gpu.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

namespace dlvk {

// Global GPU accelerator for loss functions
static std::unique_ptr<LossOpsGPU> g_loss_gpu = nullptr;
static std::shared_ptr<VulkanDevice> g_current_device = nullptr;

bool try_initialize_loss_gpu(std::shared_ptr<VulkanDevice> device) {
    if (g_loss_gpu && g_current_device == device) {
        return true; // Already initialized for this device
    }
    
    if (g_current_device != device) {
        g_loss_gpu.reset(); // Reset if device changed
        g_current_device = device;
    }
    
    if (!g_loss_gpu) {
        g_loss_gpu = std::make_unique<LossOpsGPU>(device);
        if (!g_loss_gpu->initialize()) {
            std::cout << "Warning: Failed to initialize GPU loss operations, using CPU fallback" << std::endl;
            g_loss_gpu.reset();
            return false;
        }
    }
    
    return true;
}

// Mean Squared Error Implementation
std::shared_ptr<Tensor> MeanSquaredError::forward(const std::shared_ptr<Tensor>& predictions, 
                                                 const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    auto device = predictions->device();
    auto loss = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, device);
    
    // Try GPU acceleration first
    if (try_initialize_loss_gpu(device) && g_loss_gpu->mse_forward(predictions, targets, loss)) {
        return loss;
    }
    
    // CPU Fallback implementation
    std::cout << "Using CPU fallback for MSE forward" << std::endl;
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
    
    auto device = predictions->device();
    auto gradient = std::make_shared<Tensor>(predictions->shape(), DataType::FLOAT32, device);
    
    // Try GPU acceleration first
    if (try_initialize_loss_gpu(device) && g_loss_gpu->mse_backward(predictions, targets, gradient)) {
        return gradient;
    }
    
    // CPU Fallback implementation
    std::cout << "Using CPU fallback for MSE backward" << std::endl;
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
    
    auto device = predictions->device();
    auto loss = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, device);
    
    // Try GPU acceleration first
    if (try_initialize_loss_gpu(device) && g_loss_gpu->cross_entropy_forward(predictions, targets, loss)) {
        return loss;
    }
    
    // CPU fallback
    std::cout << "Using CPU fallback for CrossEntropy forward" << std::endl;
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
    
    loss->upload_data(&total_loss);
    return loss;
}

std::shared_ptr<Tensor> CrossEntropyLoss::backward(const std::shared_ptr<Tensor>& predictions,
                                                   const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    auto device = predictions->device();
    auto gradient = std::make_shared<Tensor>(predictions->shape(), DataType::FLOAT32, device);
    
    // Try GPU acceleration first
    if (try_initialize_loss_gpu(device) && g_loss_gpu->cross_entropy_backward(predictions, targets, gradient)) {
        return gradient;
    }
    
    // CPU fallback
    std::cout << "Using CPU fallback for CrossEntropy backward" << std::endl;
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
    
    auto device = predictions->device();
    auto loss = std::make_shared<Tensor>(std::vector<size_t>{1}, DataType::FLOAT32, device);
    
    // Try GPU acceleration first
    if (try_initialize_loss_gpu(device) && g_loss_gpu->binary_cross_entropy_forward(predictions, targets, loss)) {
        return loss;
    }
    
    // CPU fallback
    std::cout << "Using CPU fallback for BinaryCrossEntropy forward" << std::endl;
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
    
    loss->upload_data(&total_loss);
    return loss;
}

std::shared_ptr<Tensor> BinaryCrossEntropyLoss::backward(const std::shared_ptr<Tensor>& predictions,
                                                         const std::shared_ptr<Tensor>& targets) {
    if (predictions->shape() != targets->shape()) {
        throw std::runtime_error("Predictions and targets must have the same shape");
    }
    
    auto device = predictions->device();
    auto gradient = std::make_shared<Tensor>(predictions->shape(), DataType::FLOAT32, device);
    
    // Try GPU acceleration first
    if (try_initialize_loss_gpu(device) && g_loss_gpu->binary_cross_entropy_backward(predictions, targets, gradient)) {
        return gradient;
    }
    
    // CPU fallback
    std::cout << "Using CPU fallback for BinaryCrossEntropy backward" << std::endl;
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
