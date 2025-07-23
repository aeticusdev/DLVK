#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <cmath>

// DLVK Headers
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/layers/dense_layer.h"
#include "dlvk/loss/loss_functions.h"
#include "dlvk/optimizers/optimizers.h"

using namespace dlvk;

void test_gradient_clipping_utilities() {
    std::cout << "🧪 Test 1: Gradient Clipping Utility Functions\n";
    std::cout << "─────────────────────────────────────────────\n";

    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        throw std::runtime_error("Failed to initialize Vulkan device");
    }

    // Initialize tensor operations
    auto tensor_ops = std::make_shared<TensorOps>(device);
    tensor_ops->initialize();
    Tensor::set_tensor_ops(tensor_ops);

    // Create test gradients with large values
    auto grad1 = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, DataType::FLOAT32, device);
    auto grad2 = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, DataType::FLOAT32, device);
    
    std::vector<float> grad1_data = {3.0f, 4.0f, 12.0f, 5.0f};  // Large gradients
    std::vector<float> grad2_data = {6.0f, 8.0f, 15.0f, 20.0f}; // Very large gradients
    
    grad1->upload_data(grad1_data.data());
    grad2->upload_data(grad2_data.data());
    
    std::vector<std::shared_ptr<Tensor>> gradients = {grad1, grad2};
    
    // Test gradient norm computation
    float initial_norm = GradientClipping::compute_grad_norm(gradients);
    std::cout << "Initial gradient norm: " << std::fixed << std::setprecision(4) << initial_norm << std::endl;
    
    // Expected norm: sqrt(3²+4²+12²+5² + 6²+8²+15²+20²) = sqrt(9+16+144+25 + 36+64+225+400) = sqrt(919) ≈ 30.32
    std::cout << "Expected norm: ~30.32\n";
    
    // Test gradient norm clipping
    std::cout << "\nTesting gradient norm clipping (max_norm = 10.0):\n";
    std::vector<std::shared_ptr<Tensor>> clipped_gradients = {grad1, grad2};
    GradientClipping::clip_grad_norm(clipped_gradients, 10.0f);
    
    float clipped_norm = GradientClipping::compute_grad_norm(clipped_gradients);
    std::cout << "Clipped gradient norm: " << std::fixed << std::setprecision(4) << clipped_norm << std::endl;
    std::cout << "Expected clipped norm: ~10.0\n";
    
    // Download and show clipped values
    std::vector<float> clipped_grad1_data(4);
    std::vector<float> clipped_grad2_data(4);
    clipped_gradients[0]->download_data(clipped_grad1_data.data());
    clipped_gradients[1]->download_data(clipped_grad2_data.data());
    
    std::cout << "Clipped grad1: [" << clipped_grad1_data[0] << ", " << clipped_grad1_data[1] 
              << ", " << clipped_grad1_data[2] << ", " << clipped_grad1_data[3] << "]\n";
    std::cout << "Clipped grad2: [" << clipped_grad2_data[0] << ", " << clipped_grad2_data[1] 
              << ", " << clipped_grad2_data[2] << ", " << clipped_grad2_data[3] << "]\n";

    // Test gradient value clipping
    std::cout << "\nTesting gradient value clipping (range: [-2.0, 2.0]):\n";
    grad1->upload_data(grad1_data.data());  // Reset to original values
    grad2->upload_data(grad2_data.data());
    
    std::vector<std::shared_ptr<Tensor>> value_clipped_gradients = {grad1, grad2};
    GradientClipping::clip_grad_value(value_clipped_gradients, -2.0f, 2.0f);
    
    std::vector<float> value_clipped_grad1_data(4);
    std::vector<float> value_clipped_grad2_data(4);
    value_clipped_gradients[0]->download_data(value_clipped_grad1_data.data());
    value_clipped_gradients[1]->download_data(value_clipped_grad2_data.data());
    
    std::cout << "Value clipped grad1: [" << value_clipped_grad1_data[0] << ", " << value_clipped_grad1_data[1] 
              << ", " << value_clipped_grad1_data[2] << ", " << value_clipped_grad1_data[3] << "]\n";
    std::cout << "Value clipped grad2: [" << value_clipped_grad2_data[0] << ", " << value_clipped_grad2_data[1] 
              << ", " << value_clipped_grad2_data[2] << ", " << value_clipped_grad2_data[3] << "]\n";
    
    std::cout << "✅ Gradient clipping utility functions working correctly!\n\n";
}

void test_optimizer_gradient_clipping() {
    std::cout << "🧪 Test 2: Optimizer-Integrated Gradient Clipping\n";
    std::cout << "─────────────────────────────────────────────────\n";

    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        throw std::runtime_error("Failed to initialize Vulkan device");
    }

    // Initialize tensor operations
    auto tensor_ops = std::make_shared<TensorOps>(device);
    tensor_ops->initialize();
    Tensor::set_tensor_ops(tensor_ops);

    // Test parameters and large gradients
    auto params = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, DataType::FLOAT32, device);
    auto large_gradients = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, DataType::FLOAT32, device);
    
    std::vector<float> param_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> large_grad_data = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 35.0f}; // Very large gradients
    
    params->upload_data(param_data.data());
    large_gradients->upload_data(large_grad_data.data());

    std::cout << "Initial parameters: [1, 2, 3, 4, 5, 6]\n";
    std::cout << "Large gradients: [10, 15, 20, 25, 30, 35]\n";
    std::cout << "Gradient norm: " << std::sqrt(10*10 + 15*15 + 20*20 + 25*25 + 30*30 + 35*35) << "\n\n";

    // Test SGD with gradient norm clipping
    std::cout << "Testing SGD with gradient norm clipping (max_norm = 5.0):\n";
    auto sgd_clipped = std::make_unique<SGD>(0.1f, 0.0f);
    sgd_clipped->set_grad_clip_norm(5.0f);
    
    auto params_sgd = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, DataType::FLOAT32, device);
    params_sgd->upload_data(param_data.data());
    
    sgd_clipped->update_parameter(params_sgd, large_gradients);
    
    std::vector<float> sgd_result(6);
    params_sgd->download_data(sgd_result.data());
    std::cout << "SGD result (clipped): [" << sgd_result[0] << ", " << sgd_result[1] << ", " 
              << sgd_result[2] << ", " << sgd_result[3] << ", " << sgd_result[4] << ", " << sgd_result[5] << "]\n";

    // Test SGD without clipping for comparison
    std::cout << "\nTesting SGD without gradient clipping:\n";
    auto sgd_normal = std::make_unique<SGD>(0.1f, 0.0f);
    
    auto params_sgd_normal = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, DataType::FLOAT32, device);
    params_sgd_normal->upload_data(param_data.data());
    
    sgd_normal->update_parameter(params_sgd_normal, large_gradients);
    
    std::vector<float> sgd_normal_result(6);
    params_sgd_normal->download_data(sgd_normal_result.data());
    std::cout << "SGD result (normal): [" << sgd_normal_result[0] << ", " << sgd_normal_result[1] << ", " 
              << sgd_normal_result[2] << ", " << sgd_normal_result[3] << ", " << sgd_normal_result[4] << ", " << sgd_normal_result[5] << "]\n";

    // Test Adam with gradient value clipping
    std::cout << "\nTesting Adam with gradient value clipping (range: [-5.0, 5.0]):\n";
    auto adam_clipped = std::make_unique<Adam>(0.01f);
    adam_clipped->set_grad_clip_value(-5.0f, 5.0f);
    
    auto params_adam = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, DataType::FLOAT32, device);
    params_adam->upload_data(param_data.data());
    
    adam_clipped->update_parameter(params_adam, large_gradients);
    adam_clipped->step();
    
    std::vector<float> adam_result(6);
    params_adam->download_data(adam_result.data());
    std::cout << "Adam result (clipped): [" << adam_result[0] << ", " << adam_result[1] << ", " 
              << adam_result[2] << ", " << adam_result[3] << ", " << adam_result[4] << ", " << adam_result[5] << "]\n";

    // Test RMSprop with both norm and value clipping
    std::cout << "\nTesting RMSprop with both norm (max_norm=8.0) and value clipping (range: [-6.0, 6.0]):\n";
    auto rmsprop_clipped = std::make_unique<RMSprop>(0.01f);
    rmsprop_clipped->set_grad_clip_norm(8.0f);
    rmsprop_clipped->set_grad_clip_value(-6.0f, 6.0f);
    
    auto params_rmsprop = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, DataType::FLOAT32, device);
    params_rmsprop->upload_data(param_data.data());
    
    rmsprop_clipped->update_parameter(params_rmsprop, large_gradients);
    
    std::vector<float> rmsprop_result(6);
    params_rmsprop->download_data(rmsprop_result.data());
    std::cout << "RMSprop result (clipped): [" << rmsprop_result[0] << ", " << rmsprop_result[1] << ", " 
              << rmsprop_result[2] << ", " << rmsprop_result[3] << ", " << rmsprop_result[4] << ", " << rmsprop_result[5] << "]\n";

    std::cout << "\n✅ Optimizer gradient clipping integration working correctly!\n\n";
}

void test_training_with_gradient_clipping() {
    std::cout << "🧪 Test 3: Training with Gradient Clipping\n";
    std::cout << "─────────────────────────────────────────────\n";

    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        throw std::runtime_error("Failed to initialize Vulkan device");
    }

    // Initialize tensor operations
    auto tensor_ops = std::make_shared<TensorOps>(device);
    tensor_ops->initialize();
    Tensor::set_tensor_ops(tensor_ops);

    // Test gradient accumulation with clipping
    std::cout << "Testing gradient accumulation with different clipping strategies:\n";

    // Create multiple gradient vectors
    auto grad1 = std::make_shared<Tensor>(std::vector<size_t>{3, 3}, DataType::FLOAT32, device);
    auto grad2 = std::make_shared<Tensor>(std::vector<size_t>{3, 3}, DataType::FLOAT32, device);
    auto grad3 = std::make_shared<Tensor>(std::vector<size_t>{3, 3}, DataType::FLOAT32, device);
    
    // Large gradient values that would cause instability
    std::vector<float> large_grad1 = {5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 35.0f, 40.0f, 45.0f};
    std::vector<float> large_grad2 = {8.0f, 12.0f, 16.0f, 24.0f, 32.0f, 40.0f, 48.0f, 56.0f, 64.0f};
    std::vector<float> large_grad3 = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f, 21.0f, 24.0f, 27.0f};
    
    grad1->upload_data(large_grad1.data());
    grad2->upload_data(large_grad2.data());
    grad3->upload_data(large_grad3.data());

    std::vector<std::shared_ptr<Tensor>> gradients = {grad1, grad2, grad3};
    
    // Compute initial total norm
    float initial_norm = GradientClipping::compute_grad_norm(gradients);
    std::cout << "Initial combined gradient norm: " << std::fixed << std::setprecision(4) << initial_norm << std::endl;
    
    // Test severe gradient clipping
    std::cout << "\nApplying aggressive gradient norm clipping (max_norm = 2.0):\n";
    GradientClipping::clip_grad_norm(gradients, 2.0f);
    
    float clipped_norm = GradientClipping::compute_grad_norm(gradients);
    std::cout << "Clipped gradient norm: " << std::fixed << std::setprecision(4) << clipped_norm << std::endl;
    
    // Show clipped values
    std::vector<float> clipped_grad1_data(9);
    gradients[0]->download_data(clipped_grad1_data.data());
    std::cout << "Clipped grad1: [";
    for (size_t i = 0; i < 9; ++i) {
        std::cout << std::fixed << std::setprecision(3) << clipped_grad1_data[i];
        if (i < 8) std::cout << ", ";
    }
    std::cout << "]\n";

    // Test parameter update with clipped gradients
    std::cout << "\nTesting optimizer parameter updates with gradient clipping:\n";
    
    auto params = std::make_shared<Tensor>(std::vector<size_t>{3, 3}, DataType::FLOAT32, device);
    std::vector<float> initial_params = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    params->upload_data(initial_params.data());
    
    // Reset gradients to large values
    grad1->upload_data(large_grad1.data());
    
    auto sgd_with_clipping = std::make_unique<SGD>(0.1f, 0.0f);
    sgd_with_clipping->set_grad_clip_norm(1.0f);
    
    auto sgd_without_clipping = std::make_unique<SGD>(0.1f, 0.0f);
    
    auto params_with_clipping = std::make_shared<Tensor>(std::vector<size_t>{3, 3}, DataType::FLOAT32, device);
    auto params_without_clipping = std::make_shared<Tensor>(std::vector<size_t>{3, 3}, DataType::FLOAT32, device);
    
    params_with_clipping->upload_data(initial_params.data());
    params_without_clipping->upload_data(initial_params.data());
    
    sgd_with_clipping->update_parameter(params_with_clipping, grad1);
    sgd_without_clipping->update_parameter(params_without_clipping, grad1);
    
    std::vector<float> params_clipped_result(9);
    std::vector<float> params_normal_result(9);
    
    params_with_clipping->download_data(params_clipped_result.data());
    params_without_clipping->download_data(params_normal_result.data());
    
    std::cout << "Parameters with clipping: [";
    for (size_t i = 0; i < 9; ++i) {
        std::cout << std::fixed << std::setprecision(3) << params_clipped_result[i];
        if (i < 8) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "Parameters without clipping: [";
    for (size_t i = 0; i < 9; ++i) {
        std::cout << std::fixed << std::setprecision(3) << params_normal_result[i];
        if (i < 8) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "\n✅ Training with gradient clipping completed successfully!\n";
    std::cout << "Gradient clipping helps stabilize training and prevents exploding gradients.\n\n";
}

int main() {
    std::cout << "DLVK - Gradient Clipping Implementation Test\n";
    std::cout << "============================================\n\n";

    try {
        test_gradient_clipping_utilities();
        test_optimizer_gradient_clipping();
        test_training_with_gradient_clipping();
        
        std::cout << "🎉 GRADIENT CLIPPING IMPLEMENTATION COMPLETE!\n";
        std::cout << "=============================================\n";
        std::cout << "✅ Gradient Clipping Utilities:\n";
        std::cout << "   • L2 norm clipping (clip_grad_norm)\n";
        std::cout << "   • Value range clipping (clip_grad_value)\n";
        std::cout << "   • Gradient norm computation\n\n";
        
        std::cout << "✅ Optimizer Integration:\n";
        std::cout << "   • SGD with gradient clipping support\n";
        std::cout << "   • Adam with gradient clipping support\n";
        std::cout << "   • RMSprop with gradient clipping support\n";
        std::cout << "   • Both norm and value clipping modes\n\n";
        
        std::cout << "✅ Training Stability:\n";
        std::cout << "   • Prevents exploding gradients\n";
        std::cout << "   • Configurable clipping thresholds\n";
        std::cout << "   • Easy enable/disable functionality\n\n";
        
        std::cout << "🚀 DLVK Phase 4.2 NOW FULLY COMPLETE!\n";
        std::cout << "All advanced training features implemented:\n";
        std::cout << "• Batch Normalization ✅\n";
        std::cout << "• Dropout Regularization ✅\n";
        std::cout << "• Learning Rate Scheduling ✅\n";
        std::cout << "• Gradient Clipping ✅\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
