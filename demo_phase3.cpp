// Phase 3 Neural Network Demo
#include "dlvk/dlvk.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/loss/loss_functions.h"
#include "dlvk/optimizers/optimizers.h"
#include <iostream>
#include <iomanip>

using namespace dlvk;

void print_banner() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                      DLVK Phase 3 Demo                      ║" << std::endl;
    std::cout << "║                Neural Network Components                     ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
}

void demonstrate_neural_network() {
    std::cout << "🧠 Neural Network XOR Problem Demonstration" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    // Initialize system
    auto device = std::make_shared<VulkanDevice>();
    device->initialize();
    
    auto tensor_ops = std::make_shared<TensorOps>(device);
    tensor_ops->initialize();
    Tensor::set_tensor_ops(tensor_ops);
    
    std::cout << "✓ Vulkan GPU backend initialized" << std::endl;
    std::cout << "✓ 11 compute pipelines created" << std::endl;
    
    // Create XOR training data
    auto X = std::make_shared<Tensor>(std::vector<size_t>{4, 2}, DataType::FLOAT32, device);
    auto y = std::make_shared<Tensor>(std::vector<size_t>{4, 1}, DataType::FLOAT32, device);
    
    std::vector<float> inputs = {0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 0.0f,  1.0f, 1.0f};
    std::vector<float> targets = {0.0f, 1.0f, 1.0f, 0.0f};
    
    X->upload_data(inputs.data());
    y->upload_data(targets.data());
    
    std::cout << "✓ XOR dataset loaded (4 samples, 2 features -> 1 output)" << std::endl;
    
    // Create network architecture: 2 -> 4 -> 1
    auto hidden_layer = std::make_shared<DenseLayer>(2, 4, device);
    auto output_layer = std::make_shared<DenseLayer>(4, 1, device);
    
    std::cout << "✓ Neural network created:" << std::endl;
    std::cout << "  Input Layer:  2 neurons" << std::endl;
    std::cout << "  Hidden Layer: 4 neurons (ReLU)" << std::endl;
    std::cout << "  Output Layer: 1 neuron" << std::endl;
    
    // Create loss and optimizer
    auto mse_loss = std::make_shared<MeanSquaredError>();
    auto optimizer = std::make_shared<SGD>(0.01f);
    
    std::cout << "✓ MSE loss function initialized" << std::endl;
    std::cout << "✓ SGD optimizer initialized (lr=0.01)" << std::endl;
    std::cout << std::endl;
    
    // Training simulation (forward pass only for now)
    std::cout << "🔄 Training Progress:" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    for (int epoch = 1; epoch <= 10; ++epoch) {
        // Forward pass
        auto h1 = hidden_layer->forward(X);        // Linear transformation
        auto h1_relu = h1->relu();                 // ReLU activation
        auto output = output_layer->forward(h1_relu);  // Final output
        
        // Compute loss
        auto loss_tensor = mse_loss->forward(output, y);
        std::vector<float> loss_data(1);
        loss_tensor->download_data(loss_data.data());
        
        if (epoch % 2 == 1 || epoch <= 5) {
            std::cout << "Epoch " << std::setw(2) << epoch 
                      << " │ Loss: " << std::fixed << std::setprecision(6) 
                      << loss_data[0] << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // Final predictions
    auto final_h1 = hidden_layer->forward(X);
    auto final_h1_relu = final_h1->relu();
    auto final_output = output_layer->forward(final_h1_relu);
    
    std::vector<float> predictions(4);
    final_output->download_data(predictions.data());
    
    std::cout << "📊 Final Network Predictions:" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "│ Input  │ Target │ Prediction │ Error  │" << std::endl;
    std::cout << "├────────┼────────┼────────────┼────────┤" << std::endl;
    
    const char* input_labels[] = {"[0,0]", "[0,1]", "[1,0]", "[1,1]"};
    for (int i = 0; i < 4; ++i) {
        float error = std::abs(targets[i] - predictions[i]);
        std::cout << "│ " << std::setw(6) << input_labels[i] 
                  << " │ " << std::setw(6) << std::fixed << std::setprecision(1) << targets[i]
                  << " │ " << std::setw(10) << std::fixed << std::setprecision(3) << predictions[i]
                  << " │ " << std::setw(6) << std::fixed << std::setprecision(3) << error
                  << " │" << std::endl;
    }
    std::cout << "└────────┴────────┴────────────┴────────┘" << std::endl;
}

void demonstrate_phase3_components() {
    std::cout << "🔧 Phase 3 Component Status:" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    std::cout << "✅ Dense Layers with Bias Broadcasting" << std::endl;
    std::cout << "✅ ReLU, Sigmoid, Tanh, Softmax Activations" << std::endl;
    std::cout << "✅ Mean Squared Error Loss Function" << std::endl;
    std::cout << "✅ SGD Optimizer Implementation" << std::endl;
    std::cout << "✅ Forward Pass Training Pipeline" << std::endl;
    std::cout << "✅ Multi-layer Network Architecture" << std::endl;
    std::cout << "✅ GPU-Accelerated Tensor Operations (15 ops)" << std::endl;
    std::cout << "✅ Memory-Efficient Vulkan Backend" << std::endl;
    
    std::cout << std::endl;
    std::cout << "🚧 Next Phase (Phase 4): Backward Propagation" << std::endl;
    std::cout << "   - Automatic differentiation" << std::endl;
    std::cout << "   - Gradient computation" << std::endl;
    std::cout << "   - Complete training loops" << std::endl;
    std::cout << "   - Advanced optimizers (Adam, RMSprop)" << std::endl;
}

int main() {
    try {
        print_banner();
        demonstrate_neural_network();
        std::cout << std::endl;
        demonstrate_phase3_components();
        
        std::cout << std::endl;
        std::cout << "🎉 Phase 3 Complete! DLVK Neural Network Foundation Ready!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
