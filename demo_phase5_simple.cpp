#include "dlvk/model/model.h"
#include "dlvk/model/callbacks.h"
#include "dlvk/layers/activation.h"
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor_ops.h"
#include <iostream>
#include <memory>
#include <map>

using namespace dlvk;

void demo_sequential_model() {
    std::cout << "\n=== Sequential Model Demo ===" << std::endl;
    
    // Create Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return;
    }
    
    // Initialize TensorOps
    if (!TensorOps::initialize(device.get())) {
        std::cerr << "Failed to initialize TensorOps" << std::endl;
        return;
    }
    
    try {
        // Create a sequential model
        Sequential model(*device);
        
        // Add layers to the model
        model.add_dense(10, 64);  // Input: 10 features, Output: 64 units
        model.add_relu();
        model.add_dense(64, 32);  // Hidden layer
        model.add_relu();
        model.add_dense(32, 1);   // Output layer
        model.add_sigmoid();
        
        // Print model summary
        std::cout << "Model created successfully!" << std::endl;
        std::cout << "Model summary:" << std::endl;
        model.summary();
        
        std::cout << "Sequential model demo completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    // Cleanup
    TensorOps::shutdown();
}

void demo_activation_layers() {
    std::cout << "\n=== Activation Layers Demo ===" << std::endl;
    
    // Create Vulkan device
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return;
    }
    
    // Initialize TensorOps
    if (!TensorOps::initialize(device.get())) {
        std::cerr << "Failed to initialize TensorOps" << std::endl;
        return;
    }
    
    try {
        std::cout << "Testing different activation layers:" << std::endl;
        
        // Test ReLU
        ActivationLayer relu_layer(*device, ActivationType::ReLU);
        std::cout << "ReLU layer info: " << relu_layer.get_layer_info().type << std::endl;
        
        // Test Sigmoid
        ActivationLayer sigmoid_layer(*device, ActivationType::Sigmoid);
        std::cout << "Sigmoid layer info: " << sigmoid_layer.get_layer_info().type << std::endl;
        
        // Test Tanh
        ActivationLayer tanh_layer(*device, ActivationType::Tanh);
        std::cout << "Tanh layer info: " << tanh_layer.get_layer_info().type << std::endl;
        
        // Test Softmax
        ActivationLayer softmax_layer(*device, ActivationType::Softmax);
        std::cout << "Softmax layer info: " << softmax_layer.get_layer_info().type << std::endl;
        
        std::cout << "Activation layers demo completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    // Cleanup
    TensorOps::shutdown();
}

void demo_training_callbacks() {
    std::cout << "\n=== Training Callbacks Demo ===" << std::endl;
    
    try {
        // Create different types of callbacks
        ProgressCallback progress_callback(true);  // Verbose progress
        EarlyStopping early_stopping("val_loss", 5, 0.001f);  // Monitor val_loss, patience 5, min_delta 0.001
        // Note: ModelCheckpoint needs a model pointer, so we'll skip it for this simple demo
        
        std::cout << "Created callbacks:" << std::endl;
        std::cout << "- Progress callback (verbose mode)" << std::endl;
        std::cout << "- Early stopping (monitor: val_loss, patience: 5, min_delta: 0.001)" << std::endl;
        
        // Simulate training epochs
        for (int epoch = 1; epoch <= 20; ++epoch) {
            float loss = 1.0f / epoch;  // Simulated decreasing loss
            float val_loss = 1.0f / epoch + 0.01f;  // Simulated validation loss
            
            TrainingMetrics metrics;
            metrics.loss = loss;
            metrics.validation_loss = val_loss;
            metrics.accuracy = 0.8f + 0.01f * epoch;
            metrics.validation_accuracy = 0.75f + 0.01f * epoch;
            metrics.epoch = epoch;
            
            // Call callbacks
            progress_callback.on_epoch_end(epoch, metrics);
            early_stopping.on_epoch_end(epoch, metrics);
            
            if (early_stopping.should_stop()) {
                std::cout << "Early stopping triggered at epoch " << epoch << std::endl;
                break;
            }
        }
        
        std::cout << "Training callbacks demo completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== DLVK Phase 5: High-Level Model APIs Demo ===" << std::endl;
    
    // Run demos
    demo_sequential_model();
    demo_activation_layers();
    demo_training_callbacks();
    
    std::cout << "\n=== All demos completed! ===" << std::endl;
    return 0;
}
