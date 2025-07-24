#pragma once

#include <memory>
#include <vector>
#include <string>
#include "dlvk/tensor/tensor.h"
#include "dlvk/layers/modern_layer.h"
#include "dlvk/optimizers/optimizers.h"
#include "dlvk/loss/loss_functions.h"

namespace dlvk {

class VulkanDevice;

class Model {
public:
    virtual ~Model() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual void backward(const Tensor& grad_output) = 0;
    virtual void update_parameters(Optimizer& optimizer) = 0;
    virtual void set_training(bool training) = 0;
    virtual std::string summary() const = 0;
    virtual size_t parameter_count() const = 0;
    virtual void save_weights(const std::string& filepath) const = 0;
    virtual void load_weights(const std::string& filepath) = 0;
};

class Sequential : public Model {
private:
    std::vector<std::unique_ptr<ModernLayer>> m_layers;
    std::shared_ptr<VulkanDevice> m_device;
    bool m_is_training = true;
    std::vector<Tensor> m_layer_outputs;

public:
    Sequential(std::shared_ptr<VulkanDevice> device);
    ~Sequential() = default;

    std::unique_ptr<Sequential> clone() const {
        auto cloned = std::make_unique<Sequential>(m_device);
        for (const auto& layer : m_layers) {
            cloned->m_layers.push_back(layer->clone());
        }
        cloned->m_is_training = m_is_training;
        return cloned;
    }

    Sequential(Sequential&& other) noexcept = default;
    Sequential& operator=(Sequential&& other) noexcept = default;
    Sequential(const Sequential&) = delete;
    Sequential& operator=(const Sequential&) = delete;

    Sequential(const Sequential& other, std::shared_ptr<VulkanDevice> device)
        : m_device(device), m_is_training(other.m_is_training) {
        for (const auto& layer : other.m_layers) {
            m_layers.push_back(layer->clone());
        }
    }

    void add_layer(std::unique_ptr<ModernLayer> layer) {
        m_layers.push_back(std::move(layer));
    }
    const std::vector<std::unique_ptr<ModernLayer>>& layers() const { return m_layers; }

    void add(std::unique_ptr<ModernLayer> layer);
    void add_dense(size_t input_size, size_t output_size, bool use_bias = true);
    void add_conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, 
                    size_t stride = 1, size_t padding = 0);
    void add_maxpool2d(size_t pool_size, size_t stride = 0);
    void add_avgpool2d(size_t pool_size, size_t stride = 0);
    void add_batchnorm1d(size_t num_features);
    void add_batchnorm2d(size_t num_features);
    void add_dropout(float dropout_rate);
    void add_relu();
    void add_sigmoid();
    void add_tanh();
    void add_softmax();

    Tensor forward(const Tensor& input) override;
    void backward(const Tensor& grad_output) override;
    void update_parameters(Optimizer& optimizer) override;
    void set_training(bool training) override;
    std::string summary() const override;
    size_t parameter_count() const override;
    void save_weights(const std::string& filepath) const override;
    void load_weights(const std::string& filepath) override;

    size_t size() const { return m_layers.size(); }
    ModernLayer& operator[](size_t index) { return *m_layers[index]; }
    const ModernLayer& operator[](size_t index) const { return *m_layers[index]; }
};

struct TrainingMetrics {
    float loss;
    float accuracy;
    float validation_loss;
    float validation_accuracy;
    size_t epoch;
    size_t batch;
    TrainingMetrics() : loss(0.0f), accuracy(0.0f), validation_loss(0.0f), 
                       validation_accuracy(0.0f), epoch(0), batch(0) {}
};

class TrainingCallback {
public:
    virtual ~TrainingCallback() = default;
    virtual void on_train_begin() {}
    virtual void on_train_end() {}
    virtual void on_epoch_begin([[maybe_unused]] size_t epoch) {}
    virtual void on_epoch_end([[maybe_unused]] size_t epoch, [[maybe_unused]] const TrainingMetrics& metrics) {}
    virtual void on_batch_begin([[maybe_unused]] size_t batch) {}
    virtual void on_batch_end([[maybe_unused]] size_t batch, [[maybe_unused]] const TrainingMetrics& metrics) {}
};

class ModelTrainer {
private:
    Model* m_model;
    std::unique_ptr<Optimizer> m_optimizer;
    std::unique_ptr<LossFunction> m_loss_function;
    std::vector<std::unique_ptr<TrainingCallback>> m_callbacks;
    
public:
    ModelTrainer(Model* model);
    ~ModelTrainer() = default;
    
    void compile(std::unique_ptr<Optimizer> optimizer, 
                std::unique_ptr<LossFunction> loss_function);
    void add_callback(std::unique_ptr<TrainingCallback> callback);
    void fit(const Tensor& x_train, const Tensor& y_train,
            size_t epochs, size_t batch_size = 32,
            float validation_split = 0.0f, bool verbose = true);
    TrainingMetrics evaluate(const Tensor& x_test, const Tensor& y_test,
                           size_t batch_size = 32);
    Tensor predict(const Tensor& x, size_t batch_size = 32);
    Optimizer* get_optimizer() { return m_optimizer.get(); }
    
private:
    float calculate_accuracy(const Tensor& predictions, const Tensor& targets);
    std::tuple<Tensor, Tensor, Tensor, Tensor> split_data(
        const Tensor& x, const Tensor& y, float validation_split);
    std::vector<std::pair<Tensor, Tensor>> create_batches(
        const Tensor& x, const Tensor& y, size_t batch_size);
};

} // namespace dlvk