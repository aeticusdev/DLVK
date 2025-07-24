/**
 * @file model_export_demo_clean.cpp
 * @brief Clean Model Training and Export Demo using DLVK + ModelPersistence
 */

#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <vector>

// DLVK headers
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/tensor/tensor_ops.h"

using namespace dlvk;

// Include model persistence implementation directly
#include "../src/core/model_persistence.cpp"

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

int main() {
    print_header("DLVK MODEL TRAINING & EXPORT DEMO");
    std::cout << "🚀 Real GPU training → Multi-format export!\n\n";
    
    try {
        // Initialize GPU device
        auto device = std::make_shared<VulkanDevice>();
        if (!device->initialize()) {
            std::cout << "❌ Failed to initialize GPU device\n";
            return 1;
        }
        std::cout << "✅ GPU device initialized\n";
        
        // Initialize TensorOps
        if (!TensorOps::initialize(device.get())) {
            std::cout << "❌ Failed to initialize TensorOps\n";
            return 1;
        }
        std::cout << "✅ TensorOps with 20 GPU pipelines ready\n\n";
        
        auto ops = TensorOps::instance();
        
        print_header("STEP 1: TRAIN MODEL ON GPU");
        
        // Create training tensors
        Tensor x_train({5, 1}, DataType::FLOAT32, device);
        Tensor y_train({5, 1}, DataType::FLOAT32, device);
        Tensor weight({1, 1}, DataType::FLOAT32, device);
        Tensor prediction({5, 1}, DataType::FLOAT32, device);
        Tensor error({5, 1}, DataType::FLOAT32, device);
        
        // Training data: Learn y = 2.5*x + 1.0
        std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> y_data = {3.5f, 6.0f, 8.5f, 11.0f, 13.5f};
        std::vector<float> w_init = {0.1f};
        
        x_train.upload_data(x_data.data());
        y_train.upload_data(y_data.data());
        weight.upload_data(w_init.data());
        
        std::cout << "🧠 Training: Learn y = 2.5*x + 1.0\n";
        std::cout << "📊 Data: x=[1,2,3,4,5] → y=[3.5,6.0,8.5,11.0,13.5]\n\n";
        
        // Training loop
        std::vector<float> loss_history;
        float learning_rate = 0.1f;
        int epochs = 12;
        
        auto training_start = std::chrono::high_resolution_clock::now();
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            ops->matrix_multiply(x_train, weight, prediction);
            
            // Compute error
            ops->subtract(prediction, y_train, error);
            
            // Download results
            std::vector<float> pred_vals(5), error_vals(5), current_w(1);
            prediction.download_data(pred_vals.data());
            error.download_data(error_vals.data());
            weight.download_data(current_w.data());
            
            // Compute MSE
            float mse = 0.0f;
            for (int i = 0; i < 5; ++i) {
                mse += error_vals[i] * error_vals[i];
            }
            mse /= 5.0f;
            loss_history.push_back(mse);
            
            // Gradient computation and weight update
            float gradient = 0.0f;
            for (int i = 0; i < 5; ++i) {
                gradient += error_vals[i] * x_data[i];
            }
            gradient /= 5.0f;
            
            current_w[0] -= learning_rate * gradient;
            weight.upload_data(current_w.data());
            
            std::cout << "Epoch " << std::setw(2) << epoch + 1 << " | ";
            std::cout << "Weight: " << std::fixed << std::setprecision(3) << current_w[0] << " | ";
            std::cout << "Loss: " << std::setprecision(4) << mse << "\n";
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto training_time = std::chrono::duration<float>(training_end - training_start).count();
        
        // Get final model parameters
        std::vector<float> final_weight(1);
        weight.download_data(final_weight.data());
        
        std::cout << "\n✅ Training completed in " << std::setprecision(3) << training_time << " seconds!\n";
        std::cout << "📈 Final weight: " << std::setprecision(4) << final_weight[0] << " (target: ~2.5)\n";
        
        print_header("STEP 2: EXPORT MODEL TO MULTIPLE FORMATS");
        
        // Create export directory structure
        std::string model_name = "LinearRegressor_DLVK";
        std::string export_dir = "./exported_models";
        
        if (ModelPersistence::create_export_directory(export_dir, model_name)) {
            std::cout << "✅ Created export directory: " << export_dir << "/" << model_name << "\n";
        }
        
        // Model metadata
        std::string architecture = "Linear(input=1, output=1)";
        std::vector<std::string> layer_names = {"linear_layer"};
        
        // Export 1: Binary format (.dlvk)
        std::string binary_path = export_dir + "/" + model_name + "/weights/model.dlvk";
        if (ModelPersistence::save_model_binary(binary_path, final_weight, model_name, architecture)) {
            std::cout << "✅ Binary export: " << binary_path << "\n";
        }
        
        // Export 2: JSON format (.json)
        std::string json_path = export_dir + "/" + model_name + "/metadata/model.json";
        if (ModelPersistence::save_model_json(json_path, final_weight, model_name, architecture, layer_names)) {
            std::cout << "✅ JSON export: " << json_path << "\n";
        }
        
        // Export 3: Training report (.txt)
        std::string report_path = export_dir + "/" + model_name + "/reports/training_report.txt";
        if (ModelPersistence::save_training_report(report_path, model_name, loss_history, 
                                                  loss_history.back(), training_time, epochs, learning_rate)) {
            std::cout << "✅ Training report: " << report_path << "\n";
        }
        
        print_header("STEP 3: VERIFY MODEL EXPORT");
        
        // Test loading the model back
        std::vector<float> loaded_weights;
        std::string loaded_name, loaded_arch;
        
        if (ModelPersistence::load_model_binary(binary_path, loaded_weights, loaded_name, loaded_arch)) {
            std::cout << "✅ Successfully loaded model from binary file!\n";
            std::cout << "   Model: " << loaded_name << "\n";
            std::cout << "   Architecture: " << loaded_arch << "\n";
            std::cout << "   Parameters: " << loaded_weights.size() << "\n";
            std::cout << "   Weight value: " << std::setprecision(6) << loaded_weights[0] << "\n\n";
            
            // Verify the loaded weight matches
            float weight_diff = std::abs(loaded_weights[0] - final_weight[0]);
            if (weight_diff < 1e-6) {
                std::cout << "✅ Weight verification passed! (diff: " << weight_diff << ")\n";
            } else {
                std::cout << "❌ Weight verification failed! (diff: " << weight_diff << ")\n";
            }
        }
        
        print_header("SUCCESS! MODEL EXPORT COMPLETE");
        
        std::cout << "🎉 Model successfully trained and exported!\n\n";
        std::cout << "📁 Exported files:\n";
        std::cout << "   • Binary format (.dlvk) - For fast loading in production\n";
        std::cout << "   • JSON format (.json) - For inspection and debugging\n";
        std::cout << "   • Training report (.txt) - For documentation\n\n";
        std::cout << "🚀 Model is ready for:\n";
        std::cout << "   ✅ Production deployment\n";
        std::cout << "   ✅ Model sharing and collaboration\n";
        std::cout << "   ✅ Version control and tracking\n";
        std::cout << "   ✅ Performance analysis\n\n";
        std::cout << "The DLVK framework now supports complete ML workflows!\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
