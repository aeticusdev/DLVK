#pragma once

#include "dlvk/layers/modern_layer.h"
#include "dlvk/layers/dense_layer.h"
#include "dlvk/layers/conv2d_layer.h"
#include "dlvk/layers/pooling_layers.h"
#include "dlvk/layers/batch_norm_layers.h"
#include "dlvk/layers/dropout_layer.h"
#include "dlvk/core/vulkan_device.h"

namespace dlvk {

/**
 * @brief Adapter for DenseLayer to work with modern API
 */
class DenseLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<DenseLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    
public:
    DenseLayerAdapter(VulkanDevice& device, size_t input_size, size_t output_size, bool use_bias = true)
        : m_device(&device), m_is_training(true) {
        m_layer = std::make_unique<DenseLayer>(device, input_size, output_size);
        m_layer->initialize_weights();
    }
    
    Tensor forward(const Tensor& input) override {
        // Convert Tensor to shared_ptr for legacy API
        auto input_ptr = std::make_shared<Tensor>(input);  // This will use our copy constructor
        auto output_ptr = m_layer->forward(input_ptr);
        return *output_ptr;  // This will also use copy constructor  
    }
    
    Tensor backward(const Tensor& grad_output) override {
        auto grad_ptr = std::make_shared<Tensor>(grad_output);
        auto result_ptr = m_layer->backward(grad_ptr);
        return *result_ptr;
    }
    
    void update_parameters(Optimizer& optimizer) override {
        // Update using the old interface
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
    
    void save_weights(std::ofstream& file) const override {
        // TODO: Implement save
    }
    
    void load_weights(std::ifstream& file) override {
        // TODO: Implement load
    }
};

/**
 * @brief Adapter for Conv2DLayer to work with modern API
 */
class Conv2DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<Conv2DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    
public:
    Conv2DLayerAdapter(VulkanDevice& device, size_t in_channels, size_t out_channels, 
                      size_t kernel_size, size_t stride = 1, size_t padding = 0)
        : m_device(&device), m_is_training(true) {
        m_layer = std::make_unique<Conv2DLayer>(device, in_channels, out_channels, 
                                               kernel_size, kernel_size, stride, stride, padding, padding);
        m_layer->initialize_weights();
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
    
    void update_parameters(Optimizer& optimizer) override {
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
    
    void save_weights(std::ofstream& file) const override {
        // TODO: Implement save
    }
    
    void load_weights(std::ifstream& file) override {
        // TODO: Implement load
    }
};

/**
 * @brief Simple adapter for pooling layers (no weights)
 */
class MaxPool2DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<MaxPool2DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    
public:
    MaxPool2DLayerAdapter(VulkanDevice& device, size_t pool_size, size_t stride = 0)
        : m_device(&device), m_is_training(true) {
        if (stride == 0) stride = pool_size;
        m_layer = std::make_unique<MaxPool2DLayer>(device, pool_size, stride);
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
    
    void update_parameters(Optimizer& optimizer) override {
        // No parameters to update
    }
    
    void set_training(bool training) override {
        m_is_training = training;
    }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "MaxPool2D";
        info.parameter_count = 0;
        info.trainable = false;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights(std::ofstream& file) const override {
        // No weights to save
    }
    
    void load_weights(std::ifstream& file) override {
        // No weights to load
    }
};

/**
 * @brief Adapter for AvgPool2DLayer to work with modern API
 */
class AvgPool2DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<AvgPool2DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    
public:
    AvgPool2DLayerAdapter(VulkanDevice& device, size_t pool_size, size_t stride = 0)
        : m_device(&device), m_is_training(true) {
        if (stride == 0) stride = pool_size;
        m_layer = std::make_unique<AvgPool2DLayer>(device, pool_size, stride);
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
    
    void update_parameters(Optimizer& optimizer) override {
        // No parameters to update
    }
    
    void set_training(bool training) override {
        m_is_training = training;
    }
    
    LayerInfo get_layer_info() const override {
        LayerInfo info;
        info.type = "AvgPool2D";
        info.parameter_count = 0;
        info.trainable = false;
        info.output_shape_str = "Unknown";
        return info;
    }
    
    void save_weights(std::ofstream& file) const override {
        // No weights to save
    }
    
    void load_weights(std::ifstream& file) override {
        // No weights to load
    }
};

/**
 * @brief Adapter for BatchNorm1DLayer to work with modern API
 */
class BatchNorm1DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<BatchNorm1DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    
public:
    BatchNorm1DLayerAdapter(VulkanDevice& device, size_t num_features)
        : m_device(&device), m_is_training(true) {
        m_layer = std::make_unique<BatchNorm1DLayer>(device, num_features);
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
    
    void update_parameters(Optimizer& optimizer) override {
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
    
    void save_weights(std::ofstream& file) const override {
        // TODO: Implement save
    }
    
    void load_weights(std::ifstream& file) override {
        // TODO: Implement load
    }
};

/**
 * @brief Adapter for BatchNorm2DLayer to work with modern API
 */
class BatchNorm2DLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<BatchNorm2DLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    
public:
    BatchNorm2DLayerAdapter(VulkanDevice& device, size_t num_features)
        : m_device(&device), m_is_training(true) {
        m_layer = std::make_unique<BatchNorm2DLayer>(device, num_features);
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
    
    void update_parameters(Optimizer& optimizer) override {
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
    
    void save_weights(std::ofstream& file) const override {
        // TODO: Implement save
    }
    
    void load_weights(std::ifstream& file) override {
        // TODO: Implement load
    }
};

/**
 * @brief Adapter for DropoutLayer to work with modern API
 */
class DropoutLayerAdapter : public ModernLayer {
private:
    std::unique_ptr<DropoutLayer> m_layer;
    VulkanDevice* m_device;
    bool m_is_training;
    
public:
    DropoutLayerAdapter(VulkanDevice& device, float dropout_rate)
        : m_device(&device), m_is_training(true) {
        m_layer = std::make_unique<DropoutLayer>(device, dropout_rate);
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
    
    void update_parameters(Optimizer& optimizer) override {
        // No parameters to update
    }
    
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
    
    void save_weights(std::ofstream& file) const override {
        // No weights to save
    }
    
    void load_weights(std::ifstream& file) override {
        // No weights to load
    }
};

} // namespace dlvk
