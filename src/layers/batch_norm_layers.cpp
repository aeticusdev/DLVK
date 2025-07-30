#include "dlvk/layers/batch_norm_layers.h"
#include "dlvk/tensor/tensor_ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace dlvk {


BatchNorm1DLayer::BatchNorm1DLayer(VulkanDevice& device, size_t num_features,
                                   float momentum, float epsilon)
    : device_(device), num_features_(num_features), momentum_(momentum), 
      epsilon_(epsilon), training_(true) {
    

    std::vector<size_t> param_shape = {num_features};
    auto device_ptr = std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){});
    
    gamma_ = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, device_ptr);
    beta_ = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, device_ptr);
    running_mean_ = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, device_ptr);
    running_var_ = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, device_ptr);
    
    initialize_parameters();
}

void BatchNorm1DLayer::initialize_parameters() {

    std::vector<float> gamma_data(num_features_, 1.0f);
    gamma_->upload_data(gamma_data.data());
    

    std::vector<float> beta_data(num_features_, 0.0f);
    beta_->upload_data(beta_data.data());
    

    std::vector<float> mean_data(num_features_, 0.0f);
    std::vector<float> var_data(num_features_, 1.0f);
    running_mean_->upload_data(mean_data.data());
    running_var_->upload_data(var_data.data());
}

std::shared_ptr<Tensor> BatchNorm1DLayer::forward(const std::shared_ptr<Tensor>& input) {
    last_input_ = input;
    

    const auto& shape = input->shape();
    if (shape.size() != 2 || shape[1] != num_features_) {
        throw std::runtime_error("BatchNorm1D expects input shape [batch_size, features]");
    }
    
    size_t batch_size = shape[0];
    size_t features = shape[1];
    

    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<float> output_data(input->size());
    std::vector<float> batch_mean(features, 0.0f);
    std::vector<float> batch_var(features, 0.0f);
    
    if (training_) {

        for (size_t f = 0; f < features; ++f) {

            float sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                sum += input_data[b * features + f];
            }
            batch_mean[f] = sum / batch_size;
            

            float var_sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = input_data[b * features + f] - batch_mean[f];
                var_sum += diff * diff;
            }
            batch_var[f] = var_sum / batch_size;
        }
        

        std::vector<float> running_mean_data(features);
        std::vector<float> running_var_data(features);
        running_mean_->download_data(running_mean_data.data());
        running_var_->download_data(running_var_data.data());
        
        for (size_t f = 0; f < features; ++f) {
            running_mean_data[f] = (1.0f - momentum_) * running_mean_data[f] + momentum_ * batch_mean[f];
            running_var_data[f] = (1.0f - momentum_) * running_var_data[f] + momentum_ * batch_var[f];
        }
        
        running_mean_->upload_data(running_mean_data.data());
        running_var_->upload_data(running_var_data.data());
    } else {

        running_mean_->download_data(batch_mean.data());
        running_var_->download_data(batch_var.data());
    }
    

    std::vector<float> gamma_data(features);
    std::vector<float> beta_data(features);
    gamma_->download_data(gamma_data.data());
    beta_->download_data(beta_data.data());
    

    std::vector<float> normalized_data(input->size());
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            size_t idx = b * features + f;
            float normalized = (input_data[idx] - batch_mean[f]) / std::sqrt(batch_var[f] + epsilon_);
            normalized_data[idx] = normalized;
            output_data[idx] = gamma_data[f] * normalized + beta_data[f];
        }
    }
    

    if (training_) {
        last_normalized_ = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, 
                                                   std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
        last_normalized_->upload_data(normalized_data.data());
    }
    

    auto output = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32,
                                          std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    output->upload_data(output_data.data());
    
    return output;
}

std::shared_ptr<Tensor> BatchNorm1DLayer::backward(const std::shared_ptr<Tensor>& grad_output) {


    return grad_output;
}

void BatchNorm1DLayer::update_weights(float learning_rate) {


}

std::unique_ptr<Layer> BatchNorm1DLayer::clone() const {
    auto cloned = std::make_unique<BatchNorm1DLayer>(device_, num_features_, momentum_, epsilon_);
    cloned->training_ = training_;
    

    if (gamma_ && cloned->gamma_) {
        std::vector<float> gamma_data(num_features_);
        gamma_->download_data(gamma_data.data());
        cloned->gamma_->upload_data(gamma_data.data());
    }
    
    if (beta_ && cloned->beta_) {
        std::vector<float> beta_data(num_features_);
        beta_->download_data(beta_data.data());
        cloned->beta_->upload_data(beta_data.data());
    }
    
    if (running_mean_ && cloned->running_mean_) {
        std::vector<float> mean_data(num_features_);
        running_mean_->download_data(mean_data.data());
        cloned->running_mean_->upload_data(mean_data.data());
    }
    
    if (running_var_ && cloned->running_var_) {
        std::vector<float> var_data(num_features_);
        running_var_->download_data(var_data.data());
        cloned->running_var_->upload_data(var_data.data());
    }
    
    return cloned;
}


BatchNorm2DLayer::BatchNorm2DLayer(VulkanDevice& device, size_t num_channels,
                                   float momentum, float epsilon)
    : device_(device), num_channels_(num_channels), momentum_(momentum), 
      epsilon_(epsilon), training_(true) {
    

    std::vector<size_t> param_shape = {num_channels};
    auto device_ptr = std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){});
    
    gamma_ = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, device_ptr);
    beta_ = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, device_ptr);
    running_mean_ = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, device_ptr);
    running_var_ = std::make_shared<Tensor>(param_shape, DataType::FLOAT32, device_ptr);
    
    initialize_parameters();
}

void BatchNorm2DLayer::initialize_parameters() {

    std::vector<float> gamma_data(num_channels_, 1.0f);
    gamma_->upload_data(gamma_data.data());
    

    std::vector<float> beta_data(num_channels_, 0.0f);
    beta_->upload_data(beta_data.data());
    

    std::vector<float> mean_data(num_channels_, 0.0f);
    std::vector<float> var_data(num_channels_, 1.0f);
    running_mean_->upload_data(mean_data.data());
    running_var_->upload_data(var_data.data());
}

std::shared_ptr<Tensor> BatchNorm2DLayer::forward(const std::shared_ptr<Tensor>& input) {
    last_input_ = input;
    

    const auto& shape = input->shape();
    if (shape.size() != 4 || shape[1] != num_channels_) {
        throw std::runtime_error("BatchNorm2D expects input shape [batch_size, channels, height, width]");
    }
    
    size_t batch_size = shape[0];
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    size_t spatial_size = height * width;
    

    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<float> output_data(input->size());
    std::vector<float> batch_mean(channels, 0.0f);
    std::vector<float> batch_var(channels, 0.0f);
    
    if (training_) {

        for (size_t c = 0; c < channels; ++c) {

            float sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        size_t idx = ((b * channels + c) * height + h) * width + w;
                        sum += input_data[idx];
                    }
                }
            }
            batch_mean[c] = sum / (batch_size * spatial_size);
            

            float var_sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        size_t idx = ((b * channels + c) * height + h) * width + w;
                        float diff = input_data[idx] - batch_mean[c];
                        var_sum += diff * diff;
                    }
                }
            }
            batch_var[c] = var_sum / (batch_size * spatial_size);
        }
        

        std::vector<float> running_mean_data(channels);
        std::vector<float> running_var_data(channels);
        running_mean_->download_data(running_mean_data.data());
        running_var_->download_data(running_var_data.data());
        
        for (size_t c = 0; c < channels; ++c) {
            running_mean_data[c] = (1.0f - momentum_) * running_mean_data[c] + momentum_ * batch_mean[c];
            running_var_data[c] = (1.0f - momentum_) * running_var_data[c] + momentum_ * batch_var[c];
        }
        
        running_mean_->upload_data(running_mean_data.data());
        running_var_->upload_data(running_var_data.data());
    } else {

        running_mean_->download_data(batch_mean.data());
        running_var_->download_data(batch_var.data());
    }
    

    std::vector<float> gamma_data(channels);
    std::vector<float> beta_data(channels);
    gamma_->download_data(gamma_data.data());
    beta_->download_data(beta_data.data());
    

    std::vector<float> normalized_data(input->size());
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t idx = ((b * channels + c) * height + h) * width + w;
                    float normalized = (input_data[idx] - batch_mean[c]) / std::sqrt(batch_var[c] + epsilon_);
                    normalized_data[idx] = normalized;
                    output_data[idx] = gamma_data[c] * normalized + beta_data[c];
                }
            }
        }
    }
    

    if (training_) {
        last_normalized_ = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, 
                                                   std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
        last_normalized_->upload_data(normalized_data.data());
    }
    

    auto output = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32,
                                          std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    output->upload_data(output_data.data());
    
    return output;
}

std::shared_ptr<Tensor> BatchNorm2DLayer::backward(const std::shared_ptr<Tensor>& grad_output) {


    return grad_output;
}

void BatchNorm2DLayer::update_weights(float learning_rate) {


}

std::unique_ptr<Layer> BatchNorm2DLayer::clone() const {
    auto cloned = std::make_unique<BatchNorm2DLayer>(device_, num_channels_, momentum_, epsilon_);
    cloned->training_ = training_;
    

    if (gamma_ && cloned->gamma_) {
        std::vector<float> gamma_data(num_channels_);
        gamma_->download_data(gamma_data.data());
        cloned->gamma_->upload_data(gamma_data.data());
    }
    
    if (beta_ && cloned->beta_) {
        std::vector<float> beta_data(num_channels_);
        beta_->download_data(beta_data.data());
        cloned->beta_->upload_data(beta_data.data());
    }
    
    if (running_mean_ && cloned->running_mean_) {
        std::vector<float> mean_data(num_channels_);
        running_mean_->download_data(mean_data.data());
        cloned->running_mean_->upload_data(mean_data.data());
    }
    
    if (running_var_ && cloned->running_var_) {
        std::vector<float> var_data(num_channels_);
        running_var_->download_data(var_data.data());
        cloned->running_var_->upload_data(var_data.data());
    }
    
    return cloned;
}

} // namespace dlvk
