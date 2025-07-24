#pragma once

#include "dlvk/layers/modern_layer.h"
#include "dlvk/layers/dense_layer.h"
#include "dlvk/layers/conv2d_layer.h"
#include "dlvk/layers/pooling_layers.h"
#include "dlvk/layers/batch_norm_layers.h"
#include "dlvk/layers/dropout_layer.h"
#include "dlvk/core/vulkan_device.h"

namespace dlvk {

class DenseLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<DenseLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    size_t m_input_size;
    size_t m_output_size;
    bool m_use_bias;
    
public:
    DenseLayerAdapter(VulkanDevice& device, size_t input_size, size_t output_size, bool use_bias = true)
        : m_device(&device), m_is_training(true), m_input_size(input_size), m_output_size(output_size), m_use_bias(use_bias) {
        m_layer = std::make_unique<DenseLayer>(device, input_size, output_size);
        m_layer->initialize_weights();
    }
    
    std::unique_ptr<ModernLayer> clone() const override {
        auto cloned = std::make_unique<DenseLayerAdapter>(*m_device, m_input_size, m_output_size, m_use_bias);
        cloned->m_is_training = m_is_training;
        // TODO: Copy weights and biases (share dense_layer.h for exact members)
        // Example: if (m_layer->get_weights()) cloned->m_layer->set_weights(std::make_shared<Tensor>(*m_layer->get_weights()));
        return cloned;
    }
    
    Tensor forward(const Tensor& input) override {
        auto input_ptr = std::make_shared<Tensor>(input);
        auto output_ptr = m_layer->forward(input_ptr);
        return *output_ptr;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        auto grad_ptr = std::make_shared<Tensor>(grad_output);
        auto result_ptr = m_layer->backward(grad_ptr);
        return *result_ptr;
    }
    
    void update_parameters([[maybe_unused]] Optimizer& optimizer) override {
        m_layer->update_weights(optimizer.get_learning_rate());
    }
    
    void set_training(bool training) override {
        m_is_training = training;
    }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "Dense";
        info.parameter_count = 0; // TODO: Calculate properly
        info.trainable = true;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights([[maybe_unused]] std::ofstream& file) const override {}
    void load_weights([[maybe_unused]] std::ifstream& file) override {}
};

class Conv2DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<Conv2DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    size_t m_in_channels;
    size_t m_out_channels;
    size_t m_kernel_size;
    size_t m_stride;
    size_t m_padding;
    
public:
    Conv2DLayerAdapter(VulkanDevice& device, size_t in_channels, size_t out_channels, 
                       size_t kernel_size, size_t stride = 1, size_t padding = 0)
        : m_device(&device), m_is_training(true), m_in_channels(in_channels), m_out_channels(out_channels),
          m_kernel_size(kernel_size), m_stride(stride), m_padding(padding) {
        m_layer = std::make_unique<Conv2DLayer>(device, in_channels, out_channels, 
                                                kernel_size, kernel_size, stride, stride, padding, padding);
        m_layer->initialize_weights();
    }
    
    std::unique_ptr<ModernLayer> clone() const override {
        auto cloned = std::make_unique<Conv2DLayerAdapter>(*m_device, m_in_channels, m_out_channels, 
                                                          m_kernel_size, m_stride, m_padding);
        cloned->m_is_training = m_is_training;
        // TODO: Copy weights and biases (share conv2d_layer.h for exact members)
        // Example: if (m_layer->get_weights()) cloned->m_layer->set_weights(std::make_shared<Tensor>(*m_layer->get_weights()));
        return cloned;
    }
    
    Tensor forward(const Tensor& input) override {
        auto input_ptr = std::make_shared<Tensor>(input);
        auto output_ptr = m_layer->forward(input_ptr);
        return *output_ptr;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        auto grad_ptr = std::make_shared<Tensor>(grad_output);
        auto result_ptr = m_layer->backward(grad_ptr);
        return *result_ptr;
    }
    
    void update_parameters([[maybe_unused]] Optimizer& optimizer) override {
        m_layer->update_weights(optimizer.get_learning_rate());
    }
    
    void set_training(bool training) override {
        m_is_training = training;
    }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "Conv2D";
        info.parameter_count = 0; // TODO: Calculate properly
        info.trainable = true;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights([[maybe_unused]] std::ofstream& file) const override {}
    void load_weights([[maybe_unused]] std::ifstream& file) override {}
};

class MaxPool2DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<MaxPool2DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    size_t m_pool_size;
    size_t m_stride;
    
public:
    MaxPool2DLayerAdapter(VulkanDevice& device, size_t pool_size, size_t stride = 0)
        : m_device(&device), m_is_training(true), m_pool_size(pool_size), m_stride(stride == 0 ? pool_size : stride) {
        m_layer = std::make_unique<MaxPool2DLayer>(device, m_pool_size, m_stride);
    }
    
    std::unique_ptr<ModernLayer> clone() const override {
        auto cloned = std::make_unique<MaxPool2DLayerAdapter>(*m_device, m_pool_size, m_stride);
        cloned->m_is_training = m_is_training;
        return cloned;
    }
    
    Tensor forward(const Tensor& input) override {
        auto input_ptr = std::make_shared<Tensor>(input);
        auto output_ptr = m_layer->forward(input_ptr);
        return *output_ptr;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        auto grad_ptr = std::make_shared<Tensor>(grad_output);
        auto result_ptr = m_layer->backward(grad_ptr);
        return *result_ptr;
    }
    
    void update_parameters([[maybe_unused]] Optimizer& optimizer) override {}
    void set_training(bool training) override { m_is_training = training; }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "MaxPool2D";
        info.parameter_count = 0;
        info.trainable = false;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights([[maybe_unused]] std::ofstream& file) const override {}
    void load_weights([[maybe_unused]] std::ifstream& file) override {}
};

class AvgPool2DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<AvgPool2DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    size_t m_pool_size;
    size_t m_stride;
    
public:
    AvgPool2DLayerAdapter(VulkanDevice& device, size_t pool_size, size_t stride = 0)
        : m_device(&device), m_is_training(true), m_pool_size(pool_size), m_stride(stride == 0 ? pool_size : stride) {
        m_layer = std::make_unique<AvgPool2DLayer>(device, m_pool_size, m_stride);
    }
    
    std::unique_ptr<ModernLayer> clone() const override {
        auto cloned = std::make_unique<AvgPool2DLayerAdapter>(*m_device, m_pool_size, m_stride);
        cloned->m_is_training = m_is_training;
        return cloned;
    }
    
    Tensor forward(const Tensor& input) override {
        auto input_ptr = std::make_shared<Tensor>(input);
        auto output_ptr = m_layer->forward(input_ptr);
        return *output_ptr;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        auto grad_ptr = std::make_shared<Tensor>(grad_output);
        auto result_ptr = m_layer->backward(grad_ptr);
        return *result_ptr;
    }
    
    void update_parameters([[maybe_unused]] Optimizer& optimizer) override {}
    void set_training(bool training) override { m_is_training = training; }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "AvgPool2D";
        info.parameter_count = 0;
        info.trainable = false;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights([[maybe_unused]] std::ofstream& file) const override {}
    void load_weights([[maybe_unused]] std::ifstream& file) override {}
};

class BatchNorm1DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<BatchNorm1DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    size_t m_num_features;
    
public:
    BatchNorm1DLayerAdapter(VulkanDevice& device, size_t num_features)
        : m_device(&device), m_is_training(true), m_num_features(num_features) {
        m_layer = std::make_unique<BatchNorm1DLayer>(device, num_features);
        m_layer->initialize_parameters(); // Ensure parameters are initialized
    }
    
    std::unique_ptr<ModernLayer> clone() const override {
        auto cloned = std::make_unique<BatchNorm1DLayerAdapter>(*m_device, m_num_features);
        cloned->m_is_training = m_is_training;
        // Copy trainable parameters using getters and setters
        if (m_layer->get_gamma()) {
            cloned->m_layer->set_gamma(std::make_shared<Tensor>(*m_layer->get_gamma()));
        }
        if (m_layer->get_beta()) {
            cloned->m_layer->set_beta(std::make_shared<Tensor>(*m_layer->get_beta()));
        }
        if (m_layer->get_running_mean()) {
            cloned->m_layer->set_running_mean(std::make_shared<Tensor>(*m_layer->get_running_mean()));
        }
        if (m_layer->get_running_var()) {
            cloned->m_layer->set_running_var(std::make_shared<Tensor>(*m_layer->get_running_var()));
        }
        return cloned;
    }
    
    Tensor forward(const Tensor& input) override {
        auto input_ptr = std::make_shared<Tensor>(input);
        auto output_ptr = m_layer->forward(input_ptr);
        return *output_ptr;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        auto grad_ptr = std::make_shared<Tensor>(grad_output);
        auto result_ptr = m_layer->backward(grad_ptr);
        return *result_ptr;
    }
    
    void update_parameters([[maybe_unused]] Optimizer& optimizer) override {
        m_layer->update_weights(optimizer.get_learning_rate());
    }
    
    void set_training(bool training) override {
        m_is_training = training;
        m_layer->set_training(training);
    }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "BatchNorm1D";
        info.parameter_count = 0; // TODO: Calculate properly
        info.trainable = true;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights([[maybe_unused]] std::ofstream& file) const override {}
    void load_weights([[maybe_unused]] std::ifstream& file) override {}
};

class BatchNorm2DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<BatchNorm2DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    size_t m_num_features;
    
public:
    BatchNorm2DLayerAdapter(VulkanDevice& device, size_t num_features)
        : m_device(&device), m_is_training(true), m_num_features(num_features) {
        m_layer = std::make_unique<BatchNorm2DLayer>(device, num_features);
        m_layer->initialize_parameters(); // Ensure parameters are initialized
    }
    
    std::unique_ptr<ModernLayer> clone() const override {
        auto cloned = std::make_unique<BatchNorm2DLayerAdapter>(*m_device, m_num_features);
        cloned->m_is_training = m_is_training;
        // Copy trainable parameters using getters and setters
        if (m_layer->get_gamma()) {
            cloned->m_layer->set_gamma(std::make_shared<Tensor>(*m_layer->get_gamma()));
        }
        if (m_layer->get_beta()) {
            cloned->m_layer->set_beta(std::make_shared<Tensor>(*m_layer->get_beta()));
        }
        if (m_layer->get_running_mean()) {
            cloned->m_layer->set_running_mean(std::make_shared<Tensor>(*m_layer->get_running_mean()));
        }
        if (m_layer->get_running_var()) {
            cloned->m_layer->set_running_var(std::make_shared<Tensor>(*m_layer->get_running_var()));
        }
        return cloned;
    }
    
    Tensor forward(const Tensor& input) override {
        auto input_ptr = std::make_shared<Tensor>(input);
        auto output_ptr = m_layer->forward(input_ptr);
        return *output_ptr;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        auto grad_ptr = std::make_shared<Tensor>(grad_output);
        auto result_ptr = m_layer->backward(grad_ptr);
        return *result_ptr;
    }
    
    void update_parameters([[maybe_unused]] Optimizer& optimizer) override {
        m_layer->update_weights(optimizer.get_learning_rate());
    }
    
    void set_training(bool training) override {
        m_is_training = training;
        m_layer->set_training(training);
    }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "BatchNorm2D";
        info.parameter_count = 0; // TODO: Calculate properly
        info.trainable = true;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights([[maybe_unused]] std::ofstream& file) const override {}
    void load_weights([[maybe_unused]] std::ifstream& file) override {}
};

class DropoutLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<DropoutLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    float m_dropout_rate;
    
public:
    DropoutLayerAdapter(VulkanDevice& device, float dropout_rate)
        : m_device(&device), m_is_training(true), m_dropout_rate(dropout_rate) {
        m_layer = std::make_unique<DropoutLayer>(device, dropout_rate);
    }
    
    std::unique_ptr<ModernLayer> clone() const override {
        auto cloned = std::make_unique<DropoutLayerAdapter>(*m_device, m_dropout_rate);
        cloned->m_is_training = m_is_training;
        return cloned;
    }
    
    Tensor forward(const Tensor& input) override {
        auto input_ptr = std::make_shared<Tensor>(input);
        auto output_ptr = m_layer->forward(input_ptr);
        return *output_ptr;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        auto grad_ptr = std::make_shared<Tensor>(grad_output);
        auto result_ptr = m_layer->backward(grad_ptr);
        return *result_ptr;
    }
    
    void update_parameters([[maybe_unused]] Optimizer& optimizer) override {}
    void set_training(bool training) override {
        m_is_training = training;
        m_layer->set_training(training);
    }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "Dropout";
        info.parameter_count = 0;
        info.trainable = false;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights([[maybe_unused]] std::ofstream& file) const override {}
    void load_weights([[maybe_unused]] std::ifstream& file) override {}
};

} // namespace dlvk