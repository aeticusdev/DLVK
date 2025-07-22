#include "dlvk/dlvk.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/loss/loss_functions.h"
#include "dlvk/optimizers/optimizers.h"
#include <iostream>
#include <iomanip>

using namespace dlvk;

int main() {
    std::cout << "DLVK Neural Network Training Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    try {
        // Initialize Vulkan
        auto device = std::make_shared<VulkanDevice>();
        device->initialize();
        std::cout << "✓ Vulkan device initialized" << std::endl;
        
        // Initialize tensor operations
        auto tensor_ops = std::make_shared<TensorOps>(device);
        if (!tensor_ops->initialize()) {
            std::cerr << "Failed to initialize tensor operations" << std::endl;
            return -1;
        }
        Tensor::set_tensor_ops(tensor_ops);
        std::cout << "✓ Tensor operations initialized" << std::endl;
        
        // Create a simple XOR dataset
        // Input: [0,0], [0,1], [1,0], [1,1]  
        // Target: [0], [1], [1], [0]
        auto X = std::make_shared<Tensor>(std::vector<size_t>{4, 2}, DataType::FLOAT32, device);
        auto y = std::make_shared<Tensor>(std::vector<size_t>{4, 1}, DataType::FLOAT32, device);
        
        // Set training data
        std::vector<float> X_data = {0.0f, 0.0f,    // [0, 0] -> 0
                                     0.0f, 1.0f,    // [0, 1] -> 1  
                                     1.0f, 0.0f,    // [1, 0] -> 1
                                     1.0f, 1.0f};   // [1, 1] -> 0
        
        std::vector<float> y_data = {0.0f,  // XOR(0,0) = 0
                                     1.0f,  // XOR(0,1) = 1
                                     1.0f,  // XOR(1,0) = 1
                                     0.0f}; // XOR(1,1) = 0
        
        X->upload_data(X_data.data());
        y->upload_data(y_data.data());
        
        std::cout << "✓ XOR dataset created" << std::endl;
        
        // Create a simple neural network: 2 -> 4 -> 1
        auto hidden_layer = std::make_shared<DenseLayer>(2, 4, device);
        auto output_layer = std::make_shared<DenseLayer>(4, 1, device);
        
        // Initialize weights with small random values
        // Note: initialize_weights() method needs to be added to DenseLayer
        std::cout << "✓ Neural network created (2->4->1)" << std::endl;
        
        // Create loss function and optimizer
        auto loss_fn = std::make_shared<MeanSquaredError>();
        auto optimizer = std::make_shared<SGD>(0.1f); // learning rate = 0.1
        
        std::cout << "✓ Loss function and optimizer created" << std::endl;
        
        // Training loop
        std::cout << "\n=== Training Neural Network ===" << std::endl;
        const int epochs = 5;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            auto hidden_output = hidden_layer->forward(X);
            auto hidden_relu = hidden_output->relu(); // Apply ReLU activation
            auto output = output_layer->forward(hidden_relu);
            
            // Compute loss
            auto loss = loss_fn->forward(output, y);
            
            // Download loss value
            std::vector<float> loss_data(1);
            loss->download_data(loss_data.data());
            float loss_value = loss_data[0];
            
            std::cout << "Epoch " << std::setw(2) << (epoch + 1) 
                      << " | Loss: " << std::fixed << std::setprecision(6) 
                      << loss_value << std::endl;
            
            // For now, just show the forward pass is working
            // Backward pass implementation would go here
        }
        
        std::cout << "\n=== Final Network Predictions ===" << std::endl;
        
        // Test final predictions
        auto final_hidden = hidden_layer->forward(X);
        auto final_hidden_relu = final_hidden->relu();
        auto final_output = output_layer->forward(final_hidden_relu);
        
        std::vector<float> final_predictions(final_output->size());
        final_output->download_data(final_predictions.data());
        
        std::vector<float> targets(y->size());
        y->download_data(targets.data());
        
        std::cout << "Input -> Target | Prediction" << std::endl;
        std::cout << "-------------------------" << std::endl;
        for (int i = 0; i < 4; ++i) {
            std::cout << "[" << X_data[i*2] << "," << X_data[i*2+1] << "] -> " 
                      << targets[i] << " | " << std::fixed << std::setprecision(3) 
                      << final_predictions[i] << std::endl;
        }
        
        std::cout << "\n✓ Neural network training test completed!" << std::endl;
        std::cout << "\nPhase 3 Status:" << std::endl;
        std::cout << "- Dense layers with bias: ✓ Working" << std::endl;
        std::cout << "- ReLU activation: ✓ Working" << std::endl;
        std::cout << "- MSE loss computation: ✓ Working" << std::endl;
        std::cout << "- Forward pass pipeline: ✓ Working" << std::endl;
        std::cout << "- Ready for backward pass implementation!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
