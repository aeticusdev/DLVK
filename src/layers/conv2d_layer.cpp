#include "dlvk/layers/conv2d_layer.h"
#include "dlvk/tensor/tensor_ops.h"
#include <random>
#include <cmath>
#include <stdexcept>

namespace dlvk {

Conv2DLayer::Conv2DLayer(VulkanDevice& device, 
                         size_t in_channels, size_t out_channels,
                         size_t kernel_height, size_t kernel_width,
                         size_t stride_h, size_t stride_w,
                         size_t padding_h, size_t padding_w)
    : device_(device), in_channels_(in_channels), out_channels_(out_channels),
      kernel_height_(kernel_height), kernel_width_(kernel_width),
      stride_h_(stride_h), stride_w_(stride_w),
      padding_h_(padding_h), padding_w_(padding_w) {
    

    std::vector<size_t> weight_shape = {out_channels, in_channels, kernel_height, kernel_width};
    weights_ = std::make_shared<Tensor>(weight_shape, DataType::FLOAT32, 
                                       std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    

    std::vector<size_t> bias_shape = {out_channels};
    bias_ = std::make_shared<Tensor>(bias_shape, DataType::FLOAT32,
                                    std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    
    initialize_weights();
}

void Conv2DLayer::initialize_weights() {

    size_t fan_in = in_channels_ * kernel_height_ * kernel_width_;
    size_t fan_out = out_channels_ * kernel_height_ * kernel_width_;
    float variance = 2.0f / (fan_in + fan_out);
    float std_dev = std::sqrt(variance);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    

    std::vector<float> weight_data(weights_->size());
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = dist(gen);
    }
    weights_->upload_data(weight_data.data());
    

    std::vector<float> bias_data(bias_->size(), 0.0f);
    bias_->upload_data(bias_data.data());
}

std::vector<size_t> Conv2DLayer::compute_output_shape(const std::vector<size_t>& input_shape) const {


    
    if (input_shape.size() != 4) {
        throw std::runtime_error("Conv2D input must be 4D: [batch, channels, height, width]");
    }
    
    size_t batch_size = input_shape[0];
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];
    
    size_t output_height = (input_height + 2 * padding_h_ - kernel_height_) / stride_h_ + 1;
    size_t output_width = (input_width + 2 * padding_w_ - kernel_width_) / stride_w_ + 1;
    
    return {batch_size, out_channels_, output_height, output_width};
}

std::shared_ptr<Tensor> Conv2DLayer::forward(const std::shared_ptr<Tensor>& input) {

    last_input_ = input;
    

    auto output_shape = compute_output_shape(input->shape());
    auto output = std::make_shared<Tensor>(output_shape, DataType::FLOAT32,
                                          std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    

    auto* tensor_ops = TensorOps::instance();
    

    tensor_ops->conv2d(*input, *weights_, *bias_, *output,
                      stride_h_, stride_w_, 
                      padding_h_, padding_w_);
    
    return output;
}

std::shared_ptr<Tensor> Conv2DLayer::backward(const std::shared_ptr<Tensor>& grad_output) {
    if (!last_input_) {
        throw std::runtime_error("Conv2DLayer::backward called without prior forward pass");
    }
    

    auto* tensor_ops = TensorOps::instance();
    

    auto input_shape = last_input_->shape();
    auto grad_input = std::make_shared<Tensor>(input_shape, DataType::FLOAT32,
                                              std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    

    if (!weight_grads_) {
        weight_grads_ = std::make_shared<Tensor>(weights_->shape(), DataType::FLOAT32,
                                                std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    }
    
    if (!bias_grads_) {
        bias_grads_ = std::make_shared<Tensor>(bias_->shape(), DataType::FLOAT32,
                                              std::shared_ptr<VulkanDevice>(&device_, [](VulkanDevice*){}));
    }
    

    tensor_ops->conv2d_backward_input(*grad_output, *weights_, *grad_input,
                                     stride_h_, stride_w_,
                                     padding_h_, padding_w_);
    

    tensor_ops->conv2d_backward_weight(*last_input_, *grad_output, *weight_grads_, *bias_grads_,
                                      stride_h_, stride_w_,
                                      padding_h_, padding_w_);
    
    return grad_input;
}

void Conv2DLayer::update_weights(float learning_rate) {
    if (!weight_grads_ || !bias_grads_) {
        throw std::runtime_error("Conv2DLayer::update_weights called without computed gradients");
    }
    

    auto* tensor_ops = TensorOps::instance();
    


    if (weight_grads_) {
        auto scaled_grads = std::make_shared<Tensor>(weight_grads_->shape(), weight_grads_->dtype(), weight_grads_->device());
        tensor_ops->scalar_multiply(*weight_grads_, learning_rate, *scaled_grads);
        tensor_ops->subtract(*weights_, *scaled_grads, *weights_);
    }
    

    if (bias_grads_ && bias_) {
        auto scaled_bias_grads = std::make_shared<Tensor>(bias_grads_->shape(), bias_grads_->dtype(), bias_grads_->device());
        tensor_ops->scalar_multiply(*bias_grads_, learning_rate, *scaled_bias_grads);
        tensor_ops->subtract(*bias_, *scaled_bias_grads, *bias_);
    }
}

std::unique_ptr<Layer> Conv2DLayer::clone() const {
    auto cloned = std::make_unique<Conv2DLayer>(device_, in_channels_, out_channels_,
                                               kernel_height_, kernel_width_,
                                               stride_h_, stride_w_,
                                               padding_h_, padding_w_);
    

    if (weights_ && cloned->weights_) {
        size_t weight_size = out_channels_ * in_channels_ * kernel_height_ * kernel_width_;
        std::vector<float> weight_data(weight_size);
        weights_->download_data(weight_data.data());
        cloned->weights_->upload_data(weight_data.data());
    }
    
    if (bias_ && cloned->bias_) {
        std::vector<float> bias_data(out_channels_);
        bias_->download_data(bias_data.data());
        cloned->bias_->upload_data(bias_data.data());
    }
    
    return cloned;
}

} // namespace dlvk
