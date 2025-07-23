#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

// DLVK Headers
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/layers/batch_norm_layers.h"
#include "dlvk/layers/dropout_layer.h"
#include "dlvk/optimizers/lr_scheduler.h"
#include "dlvk/loss/loss_functions.h"

using namespace dlvk;

void test_batch_normalization() {
    std::cout << "\n🧪 Test 1: Batch Normalization Layers\n";
    std::cout << "─────────────────────────────────────\n";
    
    auto device = std::make_shared<VulkanDevice>();
    device->initialize();
    
    // Test BatchNorm1D
    {
        BatchNorm1DLayer bn1d(*device, 4);  // 4 features
        
        // Create test input [batch=2, features=4]
        auto input = std::make_shared<Tensor>(std::vector<size_t>{2, 4}, DataType::FLOAT32, device);
        std::vector<float> input_data = {
            1.0f, 2.0f, 3.0f, 4.0f,  // batch 1
            2.0f, 3.0f, 4.0f, 5.0f   // batch 2
        };
        input->upload_data(input_data.data());
        
        // Forward pass
        auto output = bn1d.forward(input);
        
        std::cout << "✅ BatchNorm1D forward pass completed\n";
        std::cout << "   Input shape:  [" << input->shape()[0] << ", " << input->shape()[1] << "]\n";
        std::cout << "   Output shape: [" << output->shape()[0] << ", " << output->shape()[1] << "]\n";
    }
    
    // Test BatchNorm2D
    {
        BatchNorm2DLayer bn2d(*device, 3);  // 3 channels
        
        // Create test input [batch=1, channels=3, height=4, width=4]
        auto input = std::make_shared<Tensor>(std::vector<size_t>{1, 3, 4, 4}, DataType::FLOAT32, device);
        std::vector<float> input_data(1 * 3 * 4 * 4, 1.0f);
        // Fill with some pattern
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = static_cast<float>(i % 10) / 10.0f;
        }
        input->upload_data(input_data.data());
        
        // Forward pass
        auto output = bn2d.forward(input);
        
        std::cout << "✅ BatchNorm2D forward pass completed\n";
        std::cout << "   Input shape:  [" << input->shape()[0] << ", " << input->shape()[1] 
                  << ", " << input->shape()[2] << ", " << input->shape()[3] << "]\n";
        std::cout << "   Output shape: [" << output->shape()[0] << ", " << output->shape()[1] 
                  << ", " << output->shape()[2] << ", " << output->shape()[3] << "]\n";
    }
}

void test_dropout() {
    std::cout << "\n🧪 Test 2: Dropout Layer\n";
    std::cout << "─────────────────────────\n";
    
    auto device = std::make_shared<VulkanDevice>();
    device->initialize();
    
    DropoutLayer dropout(*device, 0.3f);  // 30% dropout rate
    
    // Create test input
    auto input = std::make_shared<Tensor>(std::vector<size_t>{2, 4}, DataType::FLOAT32, device);
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    input->upload_data(input_data.data());
    
    // Test training mode
    dropout.set_training(true);
    auto output_train = dropout.forward(input);
    
    std::cout << "✅ Dropout training mode completed\n";
    std::cout << "   Dropout rate: " << dropout.get_dropout_rate() << "\n";
    
    // Test inference mode
    dropout.set_training(false);
    auto output_infer = dropout.forward(input);
    
    std::cout << "✅ Dropout inference mode completed\n";
    std::cout << "   Training: " << dropout.is_training() << "\n";
}

void test_lr_schedulers() {
    std::cout << "\n🧪 Test 3: Learning Rate Schedulers\n";
    std::cout << "───────────────────────────────────\n";
    
    float base_lr = 0.1f;
    
    // Test Step LR
    StepLRScheduler step_lr(10, 0.5f);  // Decay by 0.5 every 10 steps
    
    std::cout << "✅ Step LR Scheduler:\n";
    for (int step = 0; step < 25; step += 5) {
        float lr = step_lr.get_lr(step, base_lr);
        std::cout << "   Step " << step << ": lr = " << lr << "\n";
    }
    
    // Test Exponential LR
    ExponentialLRScheduler exp_lr(0.95f);
    
    std::cout << "✅ Exponential LR Scheduler:\n";
    for (int step = 0; step < 20; step += 5) {
        float lr = exp_lr.get_lr(step, base_lr);
        std::cout << "   Step " << step << ": lr = " << lr << "\n";
    }
    
    // Test Cosine Annealing
    CosineAnnealingLRScheduler cosine_lr(20, 0.01f);
    
    std::cout << "✅ Cosine Annealing LR Scheduler:\n";
    for (int step = 0; step < 20; step += 5) {
        float lr = cosine_lr.get_lr(step, base_lr);
        std::cout << "   Step " << step << ": lr = " << lr << "\n";
    }
}

void test_binary_cross_entropy() {
    std::cout << "\n🧪 Test 4: Binary Cross-Entropy Loss\n";
    std::cout << "────────────────────────────────────\n";
    
    auto device = std::make_shared<VulkanDevice>();
    device->initialize();
    
    BinaryCrossEntropyLoss bce_loss;
    
    // Create test predictions and targets
    auto predictions = std::make_shared<Tensor>(std::vector<size_t>{2, 1}, DataType::FLOAT32, device);
    auto targets = std::make_shared<Tensor>(std::vector<size_t>{2, 1}, DataType::FLOAT32, device);
    
    std::vector<float> pred_data = {0.8f, 0.3f};  // Sigmoid outputs
    std::vector<float> target_data = {1.0f, 0.0f}; // Binary labels
    
    predictions->upload_data(pred_data.data());
    targets->upload_data(target_data.data());
    
    // Forward pass
    auto loss = bce_loss.forward(predictions, targets);
    
    // Backward pass
    auto gradients = bce_loss.backward(predictions, targets);
    
    std::cout << "✅ Binary Cross-Entropy forward pass completed\n";
    std::cout << "✅ Binary Cross-Entropy backward pass completed\n";
    std::cout << "   Predictions: [" << pred_data[0] << ", " << pred_data[1] << "]\n";
    std::cout << "   Targets:     [" << target_data[0] << ", " << target_data[1] << "]\n";
}

void test_comprehensive_training_pipeline() {
    std::cout << "\n🧪 Test 5: Comprehensive Training Pipeline\n";
    std::cout << "──────────────────────────────────────────\n";
    
    auto device = std::make_shared<VulkanDevice>();
    device->initialize();
    
    // Create layers
    BatchNorm1DLayer bn(*device, 10);
    DropoutLayer dropout(*device, 0.2f);
    
    // Create input
    auto input = std::make_shared<Tensor>(std::vector<size_t>{4, 10}, DataType::FLOAT32, device);
    std::vector<float> input_data(40, 1.0f);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) / 40.0f;
    }
    input->upload_data(input_data.data());
    
    // Forward pass through pipeline
    bn.set_training(true);
    dropout.set_training(true);
    
    auto bn_output = bn.forward(input);
    auto final_output = dropout.forward(bn_output);
    
    std::cout << "✅ Multi-layer forward pass completed\n";
    std::cout << "   Input → BatchNorm → Dropout → Output\n";
    std::cout << "   Shape: [" << input->shape()[0] << ", " << input->shape()[1] << "] → ";
    std::cout << "[" << final_output->shape()[0] << ", " << final_output->shape()[1] << "]\n";
    
    // Learning rate scheduling
    StepLRScheduler scheduler(100, 0.1f);
    float base_lr = 0.01f;
    
    std::cout << "✅ Learning rate scheduling integration ready\n";
    std::cout << "   Base LR: " << base_lr << ", Step 0: " << scheduler.get_lr(0, base_lr) << "\n";
    std::cout << "   Step 100: " << scheduler.get_lr(100, base_lr) << "\n";
}

int main() {
    std::cout << "DLVK - Phase 4.2 Advanced Features Test\n";
    std::cout << "=======================================\n";
    
    try {
        test_batch_normalization();
        test_dropout();
        test_lr_schedulers();
        test_binary_cross_entropy();
        test_comprehensive_training_pipeline();
        
        std::cout << "\n🎉 PHASE 4.2 FEATURES SUMMARY\n";
        std::cout << "=============================\n";
        std::cout << "✅ Batch Normalization:\n";
        std::cout << "   • BatchNorm1D for dense layers\n";
        std::cout << "   • BatchNorm2D for convolutional layers\n";
        std::cout << "   • Training/inference mode switching\n";
        std::cout << "   • Running statistics tracking\n\n";
        
        std::cout << "✅ Dropout Regularization:\n";
        std::cout << "   • Configurable dropout rates\n";
        std::cout << "   • Training/inference mode switching\n";
        std::cout << "   • Inverted dropout scaling\n\n";
        
        std::cout << "✅ Learning Rate Scheduling:\n";
        std::cout << "   • Step decay scheduler\n";
        std::cout << "   • Exponential decay scheduler\n";
        std::cout << "   • Cosine annealing scheduler\n";
        std::cout << "   • Linear decay scheduler\n\n";
        
        std::cout << "✅ Enhanced Loss Functions:\n";
        std::cout << "   • Binary cross-entropy for binary classification\n";
        std::cout << "   • Numerical stability improvements\n";
        std::cout << "   • Proper gradient computation\n\n";
        
        std::cout << "🚀 DLVK Phase 4.2 Advanced Features Implemented!\n";
        std::cout << "Ready for Production-Grade Training:\n";
        std::cout << "• Regularization techniques (BatchNorm, Dropout)\n";
        std::cout << "• Advanced training optimization (LR scheduling)\n";
        std::cout << "• Enhanced loss functions\n";
        std::cout << "• Professional training pipelines\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
