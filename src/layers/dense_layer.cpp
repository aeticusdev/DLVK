#include "dlvk/layers/dense_layer.h"
#include "dlvk/core/vulkan_device.h"
#include <random>
#include <iostream>

namespace dlvk {

DenseLayer::DenseLayer(VulkanDevice& device, size_t input_size, size_t output_size)
    : device_(device), input_size_(input_size), output_size_(output_size) {
    


    auto device_ptr = std::shared_ptr<VulkanDevice>(&device, [](VulkanDevice*){});
    

    weights_ = std::make_shared<Tensor>(
        std::vector<size_t>{input_size, output_size}, 
        DataType::FLOAT32, 
        device_ptr
    );
    
    bias_ = std::make_shared<Tensor>(
        std::vector<size_t>{output_size}, 
        DataType::FLOAT32, 
        device_ptr
    );
    
    initialize_weights();
}

std::shared_ptr<Tensor> DenseLayer::forward(const std::shared_ptr<Tensor>& input) {

    last_input_ = input;
    

    auto output = input->matrix_multiply(*weights_);
    

    auto result = output->add_broadcast(*bias_);
    
    return result;
}

std::shared_ptr<Tensor> DenseLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    if (!last_input_) {
        throw std::runtime_error("No forward pass recorded for backward pass");
    }
    




    

    auto weights_T = weights_->transpose();
    auto grad_input = grad_output->matrix_multiply(*weights_T);
    

    auto input_T = last_input_->transpose();
    grad_weights_ = input_T->matrix_multiply(*grad_output);
    

    grad_bias_ = grad_output->sum(0);  // Sum along batch dimension
    
    return grad_input;
}

void DenseLayer::update_weights(float learning_rate) {
    if (!grad_weights_ || !grad_bias_) {
        std::cerr << "Warning: No gradients computed for weight update" << std::endl;
        return;
    }
    

    auto lr_grad_weights = grad_weights_->multiply_scalar(-learning_rate);
    weights_ = weights_->add(*lr_grad_weights);
    

    auto lr_grad_bias = grad_bias_->multiply_scalar(-learning_rate);
    bias_ = bias_->add(*lr_grad_bias);
}

void DenseLayer::initialize_weights() {

    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (input_size_ + output_size_));
    std::normal_distribution<float> dis(0.0f, scale);
    

    std::vector<float> weight_data(input_size_ * output_size_);
    for (auto& w : weight_data) {
        w = dis(gen);
    }
    weights_->upload_data(weight_data.data());
    

    std::vector<float> bias_data(output_size_, 0.0f);
    bias_->upload_data(bias_data.data());
}

std::unique_ptr<Layer> DenseLayer::clone() const {
    auto cloned = std::make_unique<DenseLayer>(device_, input_size_, output_size_);
    

    if (weights_ && cloned->weights_) {
        std::vector<float> weight_data(input_size_ * output_size_);
        weights_->download_data(weight_data.data());
        cloned->weights_->upload_data(weight_data.data());
    }
    
    if (bias_ && cloned->bias_) {
        std::vector<float> bias_data(output_size_);
        bias_->download_data(bias_data.data());
        cloned->bias_->upload_data(bias_data.data());
    }
    
    return cloned;
}

} // namespace dlvk
