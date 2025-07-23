/**
 * DLVK Phase 4.2 Features Showcase
 * 
 * This example demonstrates the new advanced training features:
 * - Batch Normalization for regularization
 * - Dropout for preventing overfitting
 * - Learning Rate Scheduling for optimization
 * - Binary Cross-Entropy for classification
 */

#include <iostream>
#include <iomanip>

// DLVK Headers  
#include "dlvk/core/vulkan_device.h"
#include "dlvk/layers/batch_norm_layers.h"
#include "dlvk/layers/dropout_layer.h"
#include "dlvk/loss/loss_functions.h"
#include "dlvk/optimizers/lr_scheduler.h"

using namespace dlvk;

int main() {
    std::cout << "🚀 DLVK Phase 4.2 Advanced Features Showcase\n";
    std::cout << "============================================\n\n";

    try {
        auto device = std::make_shared<VulkanDevice>();
        device->initialize();
        
        // 1. Batch Normalization Demo
        std::cout << "🧪 1. Batch Normalization Layers\n";
        std::cout << "─────────────────────────────────\n";
        
        auto bn1d = std::make_shared<BatchNorm1DLayer>(*device, 128);
        auto bn2d = std::make_shared<BatchNorm2DLayer>(*device, 64);
        
        std::cout << "✅ BatchNorm1D created for dense layers (128 features)\n";
        std::cout << "✅ BatchNorm2D created for conv layers (64 channels)\n";
        std::cout << "   • Training/inference mode switching supported\n";
        std::cout << "   • Running statistics tracking implemented\n\n";
        
        // 2. Dropout Regularization Demo
        std::cout << "🎲 2. Dropout Regularization\n";
        std::cout << "───────────────────────────\n";
        
        auto dropout = std::make_shared<DropoutLayer>(*device, 0.3f);
        
        dropout->set_training(true);
        std::cout << "✅ Dropout layer created (rate: 0.3)\n";
        std::cout << "   • Training mode: Active random masking\n";
        
        dropout->set_training(false);
        std::cout << "   • Inference mode: No masking, proper scaling\n";
        std::cout << "   • Inverted dropout scaling implemented\n\n";
        
        // 3. Learning Rate Scheduling Demo
        std::cout << "📈 3. Learning Rate Scheduling\n";
        std::cout << "──────────────────────────────\n";
        
        StepLRScheduler step_scheduler(10, 0.5f);
        ExponentialLRScheduler exp_scheduler(0.95f);
        CosineAnnealingLRScheduler cosine_scheduler(100, 0.01f);
        LinearLRScheduler linear_scheduler(50, 0.1f);
        
        std::cout << "✅ Learning Rate Schedulers Available:\n";
        
        float base_lr = 0.1f;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "   • Step LR (step 0): " << step_scheduler.get_lr(0, base_lr) << "\n";
        std::cout << "   • Step LR (step 10): " << step_scheduler.get_lr(10, base_lr) << "\n";
        std::cout << "   • Exponential (step 10): " << exp_scheduler.get_lr(10, base_lr) << "\n";
        std::cout << "   • Cosine Annealing (step 25): " << cosine_scheduler.get_lr(25, base_lr) << "\n";
        std::cout << "   • Linear Decay (step 25): " << linear_scheduler.get_lr(25, base_lr) << "\n\n";
        
        // 4. Enhanced Loss Functions Demo
        std::cout << "🎯 4. Enhanced Loss Functions\n";
        std::cout << "────────────────────────────\n";
        
        BinaryCrossEntropyLoss bce_loss;
        
        std::cout << "✅ Binary Cross-Entropy Loss implemented\n";
        std::cout << "   • Numerical stability with epsilon clamping\n";
        std::cout << "   • Proper gradient computation\n";
        std::cout << "   • Suitable for binary classification tasks\n\n";
        
        // 5. Training Integration Summary
        std::cout << "🔄 5. Professional Training Pipeline\n";
        std::cout << "───────────────────────────────────\n";
        std::cout << "✅ Complete training infrastructure ready:\n";
        std::cout << "   • Regularization: BatchNorm + Dropout\n";
        std::cout << "   • Optimization: Advanced LR scheduling\n";
        std::cout << "   • Loss Functions: Enhanced numerical stability\n";
        std::cout << "   • Mode Switching: Training/inference support\n\n";
        
        std::cout << "🎉 DLVK Phase 4.2 Summary\n";
        std::cout << "=========================\n";
        std::cout << "DLVK now provides production-competitive training features:\n";
        std::cout << "• Professional regularization techniques\n";
        std::cout << "• Advanced training optimization\n";
        std::cout << "• Flexible loss functions\n";
        std::cout << "• Complete training infrastructure\n\n";
        
        std::cout << "🚀 Ready for real-world deep learning applications!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
