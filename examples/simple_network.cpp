#include <dlvk/core/vulkan_device.h>
#include <dlvk/tensor/tensor.h>
#include <dlvk/layers/layer.h>
#include <iostream>
#include <vector>

// Simple MNIST-like example
int main() {
    std::cout << "DLVK Example: Simple Neural Network\n";
    std::cout << "===================================\n\n";
    
    try {
        // Initialize Vulkan device
        auto device = std::make_shared<dlvk::VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device. Make sure you have:\n";
            std::cerr << "1. Vulkan-capable GPU\n";
            std::cerr << "2. Updated GPU drivers\n";
            std::cerr << "3. Vulkan SDK installed\n";
            return -1;
        }
        
        std::cout << "✓ Vulkan device initialized\n";
        
        // Create a simple 2-layer neural network
        // Input: 784 (28x28 image) -> Hidden: 128 -> Output: 10 (classes)
        
        const size_t batch_size = 32;
        const size_t input_size = 784;  // 28x28 MNIST images
        const size_t hidden_size = 128;
        const size_t output_size = 10;  // 10 digit classes
        
        // Create layers
        auto layer1 = std::make_shared<dlvk::DenseLayer>(input_size, hidden_size, device);
        auto layer2 = std::make_shared<dlvk::DenseLayer>(hidden_size, output_size, device);
        
        std::cout << "✓ Created neural network layers:\n";
        std::cout << "  - Input layer:  " << input_size << " neurons\n";
        std::cout << "  - Hidden layer: " << hidden_size << " neurons\n";
        std::cout << "  - Output layer: " << output_size << " neurons\n\n";
        
        // Create sample input data (simulating a batch of MNIST images)
        auto input = std::make_shared<dlvk::Tensor>(
            std::vector<size_t>{batch_size, input_size}, 
            dlvk::DataType::FLOAT32, 
            device
        );
        
        // Generate random input data (normally you'd load real images)
        std::vector<float> input_data(batch_size * input_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (auto& val : input_data) {
            val = dis(gen);
        }
        
        input->upload_data(input_data.data());
        std::cout << "✓ Generated random input data (batch size: " << batch_size << ")\n";
        
        // Forward pass simulation
        std::cout << "\nSimulating forward pass:\n";
        std::cout << "1. Input [" << batch_size << ", " << input_size << "] -> Layer 1\n";
        
        // Note: The actual matrix multiplication would be done by compute shaders
        // For now, we're just demonstrating the framework structure
        
        auto hidden = layer1->forward(input);
        std::cout << "2. Hidden [" << batch_size << ", " << hidden_size << "] -> ReLU -> Layer 2\n";
        
        // Apply ReLU activation (would be done by compute shader)
        auto hidden_relu = hidden->relu();
        
        auto output = layer2->forward(hidden_relu);
        std::cout << "3. Output [" << batch_size << ", " << output_size << "]\n";
        
        std::cout << "\n✓ Forward pass completed (structure)\n";
        
        std::cout << "\nFramework capabilities demonstrated:\n";
        std::cout << "- ✓ Vulkan device management\n";
        std::cout << "- ✓ GPU memory allocation for tensors\n";
        std::cout << "- ✓ Neural network layer structure\n";
        std::cout << "- ✓ Data flow through network\n";
        std::cout << "- ○ Compute shader execution (to be implemented)\n";
        std::cout << "- ○ Backward propagation (to be implemented)\n";
        std::cout << "- ○ Training loop (to be implemented)\n";
        
        std::cout << "\nNext development steps:\n";
        std::cout << "1. Implement compute shader dispatch system\n";
        std::cout << "2. Complete tensor operation kernels\n";
        std::cout << "3. Add backward propagation\n";
        std::cout << "4. Implement optimizers\n";
        std::cout << "5. Create training loop\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\nExample completed successfully!\n";
    return 0;
}
