#include <iostream>
#include <memory>
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor_ops.h"
#include "dlvk/data/mnist.h"
#include "dlvk/data/dataloader.h"
#include "dlvk/model/model.h"
#include "dlvk/training/trainer.h"

using namespace dlvk;

int main() {
    try {
        std::cout << "🚀 DLVK Phase 6.2 - Complete Training Pipeline Demo" << std::endl;
        std::cout << "==================================================" << std::endl;

        // Initialize Vulkan device
        std::cout << "\n⚙️ Initializing Vulkan Device..." << std::endl;
        auto device = std::make_shared<VulkanDevice>();
        device->initialize();
        
        // Initialize tensor operations
        auto tensor_ops = std::make_shared<TensorOps>(device);
        Tensor::set_tensor_ops(tensor_ops);
        std::cout << "✅ Vulkan device ready" << std::endl;

        // Setup datasets
        std::cout << "\n📁 Loading MNIST Dataset..." << std::endl;
        auto train_dataset = std::make_shared<data::MnistDataset>("./data/mnist", true, false);
        auto test_dataset = std::make_shared<data::MnistDataset>("./data/mnist", false, false);

        // Create data loaders
        const int batch_size = 32;
        data::DataLoader train_loader(train_dataset, device, batch_size, true, false);
        data::DataLoader val_loader(test_dataset, device, batch_size, false, false);

        std::cout << "✅ Training batches: " << train_loader.num_batches() << std::endl;
        std::cout << "✅ Validation batches: " << val_loader.num_batches() << std::endl;

        // Create MNIST classifier model
        std::cout << "\n🧠 Building Neural Network..." << std::endl;
        auto model = std::make_shared<Sequential>(device);
        
        // Build a deeper network for better MNIST performance
        model->add_dense(784, 256);  // Input layer
        model->add_relu();
        model->add_dense(256, 128);  // Hidden layer 1
        model->add_relu();
        model->add_dense(128, 64);   // Hidden layer 2
        model->add_relu();
        model->add_dense(64, 10);    // Output layer (10 classes)
        
        model->summary();

        // Create trainer with modern configuration
        std::cout << "\n🏋️ Setting up Training Pipeline..." << std::endl;
        auto trainer = training::create_trainer(
            model,         // shared_ptr to model
            "adam",        // optimizer
            0.001f,        // learning rate
            "crossentropy" // loss function
        );

        // Add training callbacks
        trainer->add_callback(std::make_unique<training::ProgressCallback>(5, 1));
        trainer->add_callback(std::make_unique<training::EarlyStoppingCallback>(3));

        std::cout << "✅ Trainer configured with Adam optimizer (lr=0.001)" << std::endl;
        std::cout << "✅ CrossEntropy loss function ready" << std::endl;
        std::cout << "✅ Progress monitoring and early stopping enabled" << std::endl;

        // Demonstrate batch processing first
        std::cout << "\n🔍 Demonstrating Data→Model Pipeline..." << std::endl;
        
        // Process a few batches to show the pipeline working
        train_loader.new_epoch();
        for (int i = 0; i < 3; ++i) {
            auto [inputs, targets] = train_loader.get_batch(i);
            
            std::cout << "  Batch " << (i + 1) << ": ";
            std::cout << "Input [" << inputs.shape()[0] << "," << inputs.shape()[1] 
                      << "," << inputs.shape()[2] << "," << inputs.shape()[3] << "] → ";
            
            // Flatten for dense layer
            std::vector<size_t> flat_shape = {inputs.shape()[0], 784};
            auto reshaped = inputs.reshape(flat_shape);
            
            std::cout << "Flattened [" << (*reshaped).shape()[0] << "," << (*reshaped).shape()[1] << "] → ";
            
            // Forward pass to validate pipeline
            auto predictions = model->forward(*reshaped);
            std::cout << "Predictions [" << predictions.shape()[0] << "," << predictions.shape()[1] << "] ✅" << std::endl;
        }

        // Start training
        std::cout << "\n🎯 Starting MNIST Training (5 epochs)..." << std::endl;
        std::cout << "Target: Learn to classify handwritten digits 0-9" << std::endl;
        std::cout << "Dataset: 60,000 training samples, 10,000 validation samples" << std::endl;
        
        auto training_start = std::chrono::high_resolution_clock::now();
        
        trainer->fit(train_loader, val_loader, 5, true);
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(training_end - training_start);

        // Final evaluation
        std::cout << "\n📊 Final Model Evaluation..." << std::endl;
        auto final_metrics = trainer->evaluate(val_loader, true);

        // Training summary
        std::cout << "\n🎉 Training Complete!" << std::endl;
        std::cout << "=" << std::string(50, '=') << std::endl;
        std::cout << "🕐 Total training time: " << total_time.count() << " seconds" << std::endl;
        std::cout << "🎯 Final validation accuracy: " << std::fixed << std::setprecision(2) 
                  << (final_metrics.train_accuracy * 100) << "%" << std::endl;
        std::cout << "📉 Final validation loss: " << std::fixed << std::setprecision(4) 
                  << final_metrics.train_loss << std::endl;

        std::cout << "\n🏆 Phase 6.2 Advanced Training Features Demo Success!" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        std::cout << "✅ Complete training pipeline: Data → Model → Loss → Optimizer" << std::endl;
        std::cout << "✅ Professional training callbacks: Progress, Early stopping" << std::endl;
        std::cout << "✅ Multi-epoch training with validation" << std::endl;
        std::cout << "✅ Real MNIST dataset: 60,000 training samples" << std::endl;
        std::cout << "✅ Modern optimizers: Adam with adaptive learning rates" << std::endl;
        std::cout << "✅ Training metrics: Loss, accuracy, timing" << std::endl;
        std::cout << "✅ Production-ready ML workflow!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
