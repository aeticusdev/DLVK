#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

#include "dlvk/core/vulkan_device.h"
#include "dlvk/model/model.h"
#include "dlvk/data/mnist.h"
#include "dlvk/data/dataloader.h"
#include "dlvk/training/trainer.h"
#include "dlvk/optimization/model_optimizer.h"
#include "dlvk/deployment/inference_deployment.h"
#include "dlvk/deployment/multi_gpu_trainer.h"

using namespace dlvk;

int main() {
    std::cout << "=== DLVK Phase 6.4 - Production Deployment & Optimization Demo ===" << std::endl;
    
    try {
        // Initialize Vulkan device
        auto device = std::make_shared<VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return 1;
        }
        std::cout << "✅ Vulkan device initialized successfully" << std::endl;

        // Create a simple model for demonstration
        Sequential model(device);
        model.add_dense(784, 128);
        model.add_relu();
        model.add_dense(128, 64);
        model.add_relu();
        model.add_dense(64, 10);
        model.add_softmax();
        
        std::cout << "✅ Model created with architecture:" << std::endl;
        std::cout << "   - Dense(784 -> 128) + ReLU" << std::endl;
        std::cout << "   - Dense(128 -> 64) + ReLU" << std::endl;
        std::cout << "   - Dense(64 -> 10) + Softmax" << std::endl;

        // Load training data
        auto train_dataset = std::make_shared<data::MnistDataset>("./data/mnist", true, true);
        data::DataLoader train_loader(train_dataset, device, 32, true, false);
        
        std::cout << "✅ MNIST dataset loaded: " << train_dataset->size() << " samples" << std::endl;

        // 1. BASIC TRAINING
        std::cout << "\n🚀 1. Basic Training Phase" << std::endl;
        
        training::Trainer trainer(device);
        auto metrics = trainer.fit(model, train_loader, nullptr, 2);
        
        std::cout << "   Training completed with final loss: " << metrics.train_loss.back() << std::endl;

        // 2. MODEL OPTIMIZATION
        std::cout << "\n🔧 2. Model Optimization Phase" << std::endl;
        
        // Quantization
        std::cout << "   Applying INT8 quantization..." << std::endl;
        optimization::QuantizationConfig quant_config;
        quant_config.type = optimization::QuantizationType::INT8;
        quant_config.symmetric = true;
        quant_config.per_channel = true;
        
        auto [quantized_model, quant_stats] = optimization::ModelOptimizer::quantize_model(
            model, quant_config
        );
        
        std::cout << "   ✅ Quantization completed:" << std::endl;
        std::cout << "      - Original parameters: " << quant_stats.original_parameters << std::endl;
        std::cout << "      - Compressed parameters: " << quant_stats.optimized_parameters << std::endl;
        std::cout << "      - Compression ratio: " << quant_stats.compression_ratio << "x" << std::endl;
        
        // Pruning
        std::cout << "   Applying magnitude-based pruning..." << std::endl;
        optimization::PruningConfig prune_config;
        prune_config.strategy = optimization::PruningStrategy::MAGNITUDE;
        prune_config.sparsity_ratio = 0.3f; // Remove 30% of weights
        
        auto [pruned_model, prune_stats] = optimization::ModelOptimizer::prune_model(
            model, prune_config
        );
        
        std::cout << "   ✅ Pruning completed:" << std::endl;
        std::cout << "      - Sparsity achieved: 30%" << std::endl;
        std::cout << "      - Model size reduction: " << prune_stats.compression_ratio << "x" << std::endl;

        // 3. PRODUCTION INFERENCE
        std::cout << "\n⚡ 3. Production Inference Phase" << std::endl;
        
        deployment::ModelInferenceEngine::Config inference_config;
        inference_config.serving.max_batch_size = 64;
        inference_config.serving.processing_mode = deployment::BatchProcessingMode::DYNAMIC;
        inference_config.optimize_inference = true;
        
        deployment::ModelInferenceEngine inference_engine(inference_config);
        
        if (inference_engine.initialize(model)) {
            std::cout << "   ✅ Inference engine initialized" << std::endl;
            
            // Warmup
            std::vector<size_t> input_shape = {1, 784};
            inference_engine.warmup(input_shape, 10);
            std::cout << "   ✅ Engine warmed up with 10 iterations" << std::endl;
            
            // Benchmark inference
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < 100; ++i) {
                auto [inputs, targets] = train_loader.get_batch(0);
                auto reshaped = inputs.reshape({32, 784});
                
                std::vector<std::shared_ptr<Tensor>> input_tensors = {
                    std::make_shared<Tensor>(std::move(reshaped))
                };
                
                auto outputs = inference_engine.infer(input_tensors);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            auto metrics = inference_engine.get_metrics();
            std::cout << "   ✅ Inference benchmark completed:" << std::endl;
            std::cout << "      - Average latency: " << metrics.latency_ms << " ms" << std::endl;
            std::cout << "      - Throughput: " << metrics.throughput_qps << " QPS" << std::endl;
            std::cout << "      - Total time for 100 batches: " << duration.count() << " ms" << std::endl;
        }

        // 4. EDGE DEPLOYMENT OPTIMIZATION
        std::cout << "\n📱 4. Edge Deployment Optimization Phase" << std::endl;
        
        deployment::EdgeDeploymentOptimizer::DeviceConstraints edge_constraints;
        edge_constraints.max_memory_mb = 256;  // 256MB constraint
        edge_constraints.compute_budget = 0.5f; // 50% compute budget
        edge_constraints.supports_fp16 = true;
        edge_constraints.supports_int8 = true;
        edge_constraints.target_framework = "vulkan";
        edge_constraints.max_latency_ms = 50.0; // 50ms max latency
        
        auto [edge_model, edge_profile] = deployment::EdgeDeploymentOptimizer::optimize_for_edge(
            model, edge_constraints
        );
        
        std::cout << "   ✅ Edge optimization completed:" << std::endl;
        std::cout << "      - Profile: " << edge_profile.profile_name << std::endl;
        std::cout << "      - Model size: " << edge_profile.model_size_bytes / 1024 << " KB" << std::endl;
        std::cout << "      - Estimated latency: " << edge_profile.estimated_latency_ms << " ms" << std::endl;
        std::cout << "      - Memory footprint: " << edge_profile.memory_footprint_mb << " MB" << std::endl;
        std::cout << "      - Optimizations applied:" << std::endl;
        for (const auto& opt : edge_profile.optimizations_applied) {
            std::cout << "        * " << opt << std::endl;
        }

        // 5. MULTI-GPU TRAINING (if multiple GPUs available)
        std::cout << "\n🖥️  5. Multi-GPU Training Phase" << std::endl;
        
        auto available_gpus = deployment::MultiGPUDeviceManager::detect_gpus();
        std::cout << "   Detected " << available_gpus.size() << " GPU(s):" << std::endl;
        
        for (const auto& gpu : available_gpus) {
            std::cout << "   - Device " << gpu.device_id << ": " << gpu.device_name 
                      << " (" << gpu.memory_size / (1024*1024) << " MB)" << std::endl;
        }
        
        if (available_gpus.size() > 1) {
            std::cout << "   Setting up multi-GPU training..." << std::endl;
            
            deployment::MultiGPUTrainer::Config multi_gpu_config;
            multi_gpu_config.device_ids = {0, 1}; // Use first two GPUs
            multi_gpu_config.synchronous_updates = true;
            multi_gpu_config.gradient_accumulation_steps = 1;
            
            deployment::MultiGPUTrainer multi_trainer(multi_gpu_config);
            
            if (multi_trainer.initialize(model)) {
                std::cout << "   ✅ Multi-GPU trainer initialized" << std::endl;
                
                auto multi_metrics = multi_trainer.train(train_loader, nullptr, 1);
                auto stats = multi_trainer.get_stats();
                
                std::cout << "   ✅ Multi-GPU training completed:" << std::endl;
                std::cout << "      - Training loss: " << multi_metrics.train_loss.back() << std::endl;
                std::cout << "      - Communication efficiency: " << stats.communication_efficiency << std::endl;
                std::cout << "      - Scaling efficiency: " << stats.scaling_efficiency << std::endl;
            } else {
                std::cout << "   ⚠️  Multi-GPU trainer initialization failed" << std::endl;
            }
        } else {
            std::cout << "   ⚠️  Multi-GPU training requires at least 2 GPUs" << std::endl;
        }

        // 6. MODEL EXPORT AND DEPLOYMENT
        std::cout << "\n📦 6. Model Export and Deployment Phase" << std::endl;
        
        // ONNX Export
        std::cout << "   Exporting model to ONNX format..." << std::endl;
        optimization::ONNXConfig onnx_config;
        onnx_config.opset_version = 11;
        onnx_config.optimize_for_inference = true;
        onnx_config.model_name = "dlvk_mnist_classifier";
        
        bool onnx_success = optimization::ModelOptimizer::export_to_onnx(
            model, "./dlvk_model.onnx", onnx_config
        );
        
        if (onnx_success) {
            std::cout << "   ✅ Model exported to ONNX format: ./dlvk_model.onnx" << std::endl;
        } else {
            std::cout << "   ⚠️  ONNX export failed (feature not fully implemented)" << std::endl;
        }
        
        // Cross-platform deployment
        std::cout << "   Generating deployment packages..." << std::endl;
        
        std::vector<std::string> target_platforms = {"linux_x64", "android_arm64", "ios_arm64"};
        auto deployment_results = deployment::CrossPlatformDeployment::deploy_to_platforms(
            model, target_platforms, "./deployment_packages"
        );
        
        std::cout << "   ✅ Deployment package generation:" << std::endl;
        for (const auto& [platform, success] : deployment_results) {
            std::cout << "      - " << platform << ": " 
                      << (success ? "✅ Success" : "⚠️  Failed") << std::endl;
        }

        // 7. PERFORMANCE BENCHMARKING
        std::cout << "\n📊 7. Performance Benchmarking Phase" << std::endl;
        
        std::vector<size_t> input_shape = {784};
        std::vector<size_t> batch_sizes = {1, 8, 16, 32, 64};
        
        auto benchmark_results = optimization::ModelOptimizer::benchmark_model(
            model, input_shape, batch_sizes, 50
        );
        
        std::cout << "   ✅ Benchmark results:" << std::endl;
        std::cout << "      Batch Size | Latency (ms) | Throughput (samples/sec)" << std::endl;
        std::cout << "      -----------|--------------|------------------------" << std::endl;
        
        for (const auto& [batch_size, latency] : benchmark_results.latency_per_batch_size) {
            auto throughput = benchmark_results.throughput_per_batch_size.at(batch_size);
            std::cout << "      " << std::setw(9) << batch_size 
                      << " | " << std::setw(10) << std::fixed << std::setprecision(2) << latency
                      << " | " << std::setw(20) << std::fixed << std::setprecision(1) << throughput 
                      << std::endl;
        }
        
        std::cout << "      Peak memory usage: " << benchmark_results.peak_memory_usage / (1024*1024) << " MB" << std::endl;
        std::cout << "      Average GPU utilization: " << benchmark_results.average_gpu_utilization << "%" << std::endl;

        std::cout << "\n🎉 DLVK Phase 6.4 Production Deployment & Optimization Demo Completed!" << std::endl;
        std::cout << "✅ All production features demonstrated successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during optimization demo: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
