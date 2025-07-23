#pragma once

#include <memory>
#include <vector>
#include <string>
#include "dlvk/tensor/tensor.h"
#include "dlvk/layers/modern_layer.h"
#include "dlvk/optimizers/optimizers.h"
#include "dlvk/loss/loss_functions.h"

namespace dlvk {

// Forward declarations
class VulkanDevice;

/**
 * @brief Abstract base class for all models
 */
class Model {
public:
    virtual ~Model() = default;
    
    /**
     * @brief Forward pass through the model
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * @brief Backward pass through the model
     * @param grad_output Gradient of the loss with respect to output
     */
    virtual void backward(const Tensor& grad_output) = 0;
    
    /**
     * @brief Update model parameters using the provided optimizer
     * @param optimizer Optimizer to use for parameter updates
     */
    virtual void update_parameters(Optimizer& optimizer) = 0;
    
    /**
     * @brief Set the model to training mode
     * @param training Whether the model should be in training mode
     */
    virtual void set_training(bool training) = 0;
    
    /**
     * @brief Get a summary of the model architecture
     * @return String representation of the model
     */
    virtual std::string summary() const = 0;
    
    /**
     * @brief Get the total number of parameters in the model
     * @return Number of trainable parameters
     */
    virtual size_t parameter_count() const = 0;
    
    /**
     * @brief Save model weights to a file
     * @param filepath Path to save the weights
     */
    virtual void save_weights(const std::string& filepath) const = 0;
    
    /**
     * @brief Load model weights from a file
     * @param filepath Path to load the weights from
     */
    virtual void load_weights(const std::string& filepath) = 0;
};

/**
 * @brief Sequential model for building neural networks layer by layer
 */
class Sequential : public Model {
private:
    std::vector<std::unique_ptr<ModernLayer>> m_layers;
    std::vector<Tensor> m_layer_outputs;
    bool m_is_training;
    VulkanDevice* m_device;
    
public:
    Sequential(VulkanDevice& device);
    ~Sequential() override = default;
    
    /**
     * @brief Add a layer to the sequential model
     * @param layer Unique pointer to the layer to add
     */
    void add(std::unique_ptr<ModernLayer> layer);
    
    /**
     * @brief Add a Dense/Linear layer
     * @param input_size Number of input features
     * @param output_size Number of output features
     * @param use_bias Whether to use bias parameters
     */
    void add_dense(size_t input_size, size_t output_size, bool use_bias = true);
    
    /**
     * @brief Add a Conv2D layer
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Size of the convolution kernel
     * @param stride Stride of the convolution
     * @param padding Padding applied to input
     */
    void add_conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, 
                   size_t stride = 1, size_t padding = 0);
    
    /**
     * @brief Add a MaxPool2D layer
     * @param pool_size Size of the pooling window
     * @param stride Stride of the pooling operation
     */
    void add_maxpool2d(size_t pool_size, size_t stride = 0);
    
    /**
     * @brief Add an AvgPool2D layer
     * @param pool_size Size of the pooling window
     * @param stride Stride of the pooling operation
     */
    void add_avgpool2d(size_t pool_size, size_t stride = 0);
    
    /**
     * @brief Add a BatchNorm1D layer
     * @param num_features Number of features to normalize
     */
    void add_batchnorm1d(size_t num_features);
    
    /**
     * @brief Add a BatchNorm2D layer
     * @param num_features Number of channels to normalize
     */
    void add_batchnorm2d(size_t num_features);
    
    /**
     * @brief Add a Dropout layer
     * @param dropout_rate Probability of setting elements to zero
     */
    void add_dropout(float dropout_rate);
    
    /**
     * @brief Add a ReLU activation layer
     */
    void add_relu();
    
    /**
     * @brief Add a Sigmoid activation layer
     */
    void add_sigmoid();
    
    /**
     * @brief Add a Tanh activation layer
     */
    void add_tanh();
    
    /**
     * @brief Add a Softmax activation layer
     */
    void add_softmax();
    
    // Override virtual methods
    Tensor forward(const Tensor& input) override;
    void backward(const Tensor& grad_output) override;
    void update_parameters(Optimizer& optimizer) override;
    void set_training(bool training) override;
    std::string summary() const override;
    size_t parameter_count() const override;
    void save_weights(const std::string& filepath) const override;
    void load_weights(const std::string& filepath) override;
    
    /**
     * @brief Get the number of layers in the model
     * @return Number of layers
     */
    size_t size() const { return m_layers.size(); }
    
    /**
     * @brief Get a reference to a specific layer
     * @param index Index of the layer
     * @return Reference to the layer
     */
    ModernLayer& operator[](size_t index) { return *m_layers[index]; }
    const ModernLayer& operator[](size_t index) const { return *m_layers[index]; }
};

/**
 * @brief Training metrics for monitoring model performance
 */
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

/**
 * @brief Callback interface for training events
 */
class TrainingCallback {
public:
    virtual ~TrainingCallback() = default;
    
    /**
     * @brief Called at the beginning of training
     */
    virtual void on_train_begin() {}
    
    /**
     * @brief Called at the end of training
     */
    virtual void on_train_end() {}
    
    /**
     * @brief Called at the beginning of each epoch
     * @param epoch Current epoch number
     */
    virtual void on_epoch_begin(size_t epoch) {}
    
    /**
     * @brief Called at the end of each epoch
     * @param epoch Current epoch number
     * @param metrics Training metrics for this epoch
     */
    virtual void on_epoch_end(size_t epoch, const TrainingMetrics& metrics) {}
    
    /**
     * @brief Called at the beginning of each batch
     * @param batch Current batch number
     */
    virtual void on_batch_begin(size_t batch) {}
    
    /**
     * @brief Called at the end of each batch
     * @param batch Current batch number
     * @param metrics Training metrics for this batch
     */
    virtual void on_batch_end(size_t batch, const TrainingMetrics& metrics) {}
};

/**
 * @brief Model trainer for automating the training process
 */
class ModelTrainer {
private:
    Model* m_model;
    std::unique_ptr<Optimizer> m_optimizer;
    std::unique_ptr<LossFunction> m_loss_function;
    std::vector<std::unique_ptr<TrainingCallback>> m_callbacks;
    
public:
    ModelTrainer(Model* model);
    ~ModelTrainer() = default;
    
    /**
     * @brief Compile the model with optimizer and loss function
     * @param optimizer Optimizer for training
     * @param loss_function Loss function to minimize
     */
    void compile(std::unique_ptr<Optimizer> optimizer, 
                std::unique_ptr<LossFunction> loss_function);
    
    /**
     * @brief Add a training callback
     * @param callback Callback to add
     */
    void add_callback(std::unique_ptr<TrainingCallback> callback);
    
    /**
     * @brief Train the model on the provided data
     * @param x_train Training input data
     * @param y_train Training target data
     * @param epochs Number of training epochs
     * @param batch_size Size of training batches
     * @param validation_split Fraction of data to use for validation
     * @param verbose Whether to print training progress
     */
    void fit(const Tensor& x_train, const Tensor& y_train,
            size_t epochs, size_t batch_size = 32,
            float validation_split = 0.0f, bool verbose = true);
    
    /**
     * @brief Evaluate the model on test data
     * @param x_test Test input data
     * @param y_test Test target data
     * @param batch_size Size of evaluation batches
     * @return Evaluation metrics
     */
    TrainingMetrics evaluate(const Tensor& x_test, const Tensor& y_test,
                           size_t batch_size = 32);
    
    /**
     * @brief Make predictions on new data
     * @param x Input data for prediction
     * @param batch_size Size of prediction batches
     * @return Predictions
     */
    Tensor predict(const Tensor& x, size_t batch_size = 32);
    
    /**
     * @brief Get the optimizer (for callbacks)
     * @return Pointer to the optimizer
     */
    Optimizer* get_optimizer() { return m_optimizer.get(); }
    
private:
    /**
     * @brief Calculate accuracy for classification tasks
     * @param predictions Model predictions
     * @param targets True targets
     * @return Accuracy as a fraction between 0 and 1
     */
    float calculate_accuracy(const Tensor& predictions, const Tensor& targets);
    
    /**
     * @brief Split data into training and validation sets
     * @param x Input data
     * @param y Target data
     * @param validation_split Fraction for validation
     * @return Tuple of (x_train, y_train, x_val, y_val)
     */
    std::tuple<Tensor, Tensor, Tensor, Tensor> split_data(
        const Tensor& x, const Tensor& y, float validation_split);
    
    /**
     * @brief Create batches from data
     * @param x Input data
     * @param y Target data
     * @param batch_size Size of each batch
     * @return Vector of (x_batch, y_batch) pairs
     */
    std::vector<std::pair<Tensor, Tensor>> create_batches(
        const Tensor& x, const Tensor& y, size_t batch_size);
};

} // namespace dlvk
