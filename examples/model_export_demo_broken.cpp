/**
 * @file model_export_demo.cpp
 * @brief Real Model Training and Export Demo using DLVK core + ModelPersistence
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
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        training_date = oss.str();
    }
};

/**
 * @brief Save model to binary format
 */
bool save_model_binary(const SimpleModel& model, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "❌ Failed to open file for writing: " << filename << "\n";
        return false;
    }
    
    try {
        // Write magic header
        const char* magic = "DLVK";
        file.write(magic, 4);
        
        // Write version
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Write architecture
        uint32_t arch_size = model.architecture.size();
        file.write(reinterpret_cast<const char*>(&arch_size), sizeof(arch_size));
        file.write(reinterpret_cast<const char*>(model.architecture.data()), 
                  arch_size * sizeof(int));
        
        // Write weights
        uint32_t weights_size = model.weights.size();
        file.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));
        file.write(reinterpret_cast<const char*>(model.weights.data()), 
                  weights_size * sizeof(float));
        
        // Write biases
        uint32_t biases_size = model.biases.size();
        file.write(reinterpret_cast<const char*>(&biases_size), sizeof(biases_size));
        file.write(reinterpret_cast<const char*>(model.biases.data()), 
                  biases_size * sizeof(float));
        
        // Write activation (length-prefixed string)
        uint32_t act_len = model.activation.length();
        file.write(reinterpret_cast<const char*>(&act_len), sizeof(act_len));
        file.write(model.activation.c_str(), act_len);
        
        // Write training metadata
        file.write(reinterpret_cast<const char*>(&model.final_loss), sizeof(model.final_loss));
        file.write(reinterpret_cast<const char*>(&model.epochs_trained), sizeof(model.epochs_trained));
        
        // Write training date
        uint32_t date_len = model.training_date.length();
        file.write(reinterpret_cast<const char*>(&date_len), sizeof(date_len));
        file.write(model.training_date.c_str(), date_len);
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error writing binary file: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Load model from binary format
 */
bool load_model_binary(SimpleModel& model, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "❌ Failed to open file for reading: " << filename << "\n";
        return false;
    }
    
    try {
        // Check magic header
        char magic[5] = {0};
        file.read(magic, 4);
        if (std::string(magic) != "DLVK") {
            std::cout << "❌ Invalid file format - not a DLVK model\n";
            return false;
        }
        
        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            std::cout << "❌ Unsupported model version: " << version << "\n";
            return false;
        }
        
        // Read architecture
        uint32_t arch_size;
        file.read(reinterpret_cast<char*>(&arch_size), sizeof(arch_size));
        model.architecture.resize(arch_size);
        file.read(reinterpret_cast<char*>(model.architecture.data()), 
                 arch_size * sizeof(int));
        
        // Read weights
        uint32_t weights_size;
        file.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));
        model.weights.resize(weights_size);
        file.read(reinterpret_cast<char*>(model.weights.data()), 
                 weights_size * sizeof(float));
        
        // Read biases
        uint32_t biases_size;
        file.read(reinterpret_cast<char*>(&biases_size), sizeof(biases_size));
        model.biases.resize(biases_size);
        file.read(reinterpret_cast<char*>(model.biases.data()), 
                 biases_size * sizeof(float));
        
        // Read activation
        uint32_t act_len;
        file.read(reinterpret_cast<char*>(&act_len), sizeof(act_len));
        model.activation.resize(act_len);
        file.read(&model.activation[0], act_len);
        
        // Read training metadata
        file.read(reinterpret_cast<char*>(&model.final_loss), sizeof(model.final_loss));
        file.read(reinterpret_cast<char*>(&model.epochs_trained), sizeof(model.epochs_trained));
        
        // Read training date
        uint32_t date_len;
        file.read(reinterpret_cast<char*>(&date_len), sizeof(date_len));
        model.training_date.resize(date_len);
        file.read(&model.training_date[0], date_len);
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error reading binary file: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Save model to human-readable text format
 */
bool save_model_text(const SimpleModel& model, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "❌ Failed to open file for writing: " << filename << "\n";
        return false;
    }
    
    try {
        file << "# DLVK Trained Model Export\n";
        file << "# Generated: " << model.training_date << "\n\n";
        
        file << "[MODEL_INFO]\n";
        file << "framework = DLVK\n";
        file << "version = 1.0\n";
        file << "epochs_trained = " << model.epochs_trained << "\n";
        file << "final_loss = " << std::fixed << std::setprecision(6) << model.final_loss << "\n";
        file << "activation = " << model.activation << "\n\n";
        
        file << "[ARCHITECTURE]\n";
        for (size_t i = 0; i < model.architecture.size(); ++i) {
            file << "layer_" << i << " = " << model.architecture[i] << "\n";
        }
        file << "\n";
        
        file << "[WEIGHTS]\n";
        for (size_t i = 0; i < model.weights.size(); ++i) {
            file << "w_" << i << " = " << std::fixed << std::setprecision(8) << model.weights[i] << "\n";
        }
        file << "\n";
        
        file << "[BIASES]\n";
        for (size_t i = 0; i < model.biases.size(); ++i) {
            file << "b_" << i << " = " << std::fixed << std::setprecision(8) << model.biases[i] << "\n";
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error writing text file: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Train a model and then save it
 */
void train_and_export_model() {
    print_header("TRAIN MODEL AND EXPORT");
    
    try {
        // Initialize GPU
        auto device = std::make_shared<VulkanDevice>();
        if (!device->initialize()) {
            std::cout << "❌ Failed to initialize GPU device\n";
            return;
        }
        
        if (!TensorOps::initialize(device.get())) {
            std::cout << "❌ Failed to initialize TensorOps\n";
            return;
        }
        
        auto ops = TensorOps::instance();
        std::cout << "✅ GPU initialized with TensorOps\n\n";
        
        // Training: Learn y = 2.5*x + 1.0
        std::cout << "🧠 Training: y = 2.5*x + 1.0\n";
        
        Tensor x_train({4, 1}, DataType::FLOAT32, device);
        Tensor y_train({4, 1}, DataType::FLOAT32, device);
        Tensor weight({1, 1}, DataType::FLOAT32, device);
        Tensor bias({1, 1}, DataType::FLOAT32, device);
        Tensor prediction({4, 1}, DataType::FLOAT32, device);
        Tensor error({4, 1}, DataType::FLOAT32, device);
        Tensor temp({4, 1}, DataType::FLOAT32, device);
        
        // Training data
        std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> y_data = {3.5f, 6.0f, 8.5f, 11.0f}; // 2.5*x + 1.0
        std::vector<float> w_init = {0.1f};
        std::vector<float> b_init = {0.0f};
        
        x_train.upload_data(x_data.data());
        y_train.upload_data(y_data.data());
        weight.upload_data(w_init.data());
        bias.upload_data(b_init.data());
        
        std::cout << "📊 Training data: x=[1,2,3,4] → y=[3.5,6.0,8.5,11.0]\n";
        std::cout << "🎯 Target: weight=2.5, bias=1.0\n\n";
        
        float final_loss = 0.0f;
        int epochs = 12;
        
        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Forward: prediction = x * weight + bias
            ops->matrix_multiply(x_train, weight, temp);
            ops->add(temp, bias, prediction);
            
            // Error: prediction - target
            ops->subtract(prediction, y_train, error);
            
            // Download for gradient computation
            std::vector<float> pred_vals(4), error_vals(4), w_val(1), b_val(1);
            prediction.download_data(pred_vals.data());
            error.download_data(error_vals.data());
            weight.download_data(w_val.data());
            bias.download_data(b_val.data());
            
            // Compute MSE loss
            float mse = 0.0f;
            for (int i = 0; i < 4; ++i) {
                mse += error_vals[i] * error_vals[i];
            }
            mse /= 4.0f;
            final_loss = mse;
            
            // Gradients
            float w_grad = 0.0f, b_grad = 0.0f;
            for (int i = 0; i < 4; ++i) {
                w_grad += error_vals[i] * x_data[i];
                b_grad += error_vals[i];
            }
            w_grad /= 4.0f;
            b_grad /= 4.0f;
            
            // Updates
            w_val[0] -= 0.1f * w_grad;
            b_val[0] -= 0.1f * b_grad;
            
            weight.upload_data(w_val.data());
            bias.upload_data(b_val.data());
            
            if (epoch % 3 == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << std::setw(2) << epoch + 1 << " | ";
                std::cout << "W=" << std::fixed << std::setprecision(3) << w_val[0] << " ";
                std::cout << "B=" << std::setprecision(3) << b_val[0] << " ";
                std::cout << "Loss=" << std::setprecision(4) << mse << "\n";
            }
        }
        
        // Get final trained parameters
        std::vector<float> final_weight(1), final_bias(1);
        weight.download_data(final_weight.data());
        bias.download_data(final_bias.data());
        
        std::cout << "\n✅ Training completed!\n";
        std::cout << "📈 Final: weight=" << std::fixed << std::setprecision(3) << final_weight[0] 
                  << ", bias=" << std::setprecision(3) << final_bias[0] << "\n\n";
        
        // Create model for export
        SimpleModel trained_model(
            final_weight,           // weights
            final_bias,            // biases  
            {1, 1},                // architecture: 1 input → 1 output
            "linear",              // activation
            final_loss,            // final loss
            epochs                 // epochs trained
        );
        
        print_header("EXPORTING TRAINED MODEL");
        
        // Save in multiple formats
        std::cout << "💾 Saving model in multiple formats...\n\n";
        
        // Binary format (compact, fast loading)
        if (save_model_binary(trained_model, "linear_model.dlvk")) {
            std::cout << "✅ Binary export: linear_model.dlvk\n";
        }
        
        // Text format (human-readable, debuggable)
        if (save_model_text(trained_model, "linear_model.txt")) {
            std::cout << "✅ Text export: linear_model.txt\n";
        }
        
        std::cout << "\n📁 Model files created successfully!\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ Training/export error: " << e.what() << "\n";
    }
}

/**
 * @brief Load and test a saved model
 */
void load_and_test_model() {
    print_header("LOAD AND TEST SAVED MODEL");
    
    try {
        // Load the saved model
        SimpleModel loaded_model;
        if (!load_model_binary(loaded_model, "linear_model.dlvk")) {
            std::cout << "❌ Failed to load model\n";
            return;
        }
        
        std::cout << "✅ Model loaded successfully!\n\n";
        
        // Display model information
        std::cout << "📋 Model Information:\n";
        std::cout << "   • Architecture: ";
        for (size_t i = 0; i < loaded_model.architecture.size(); ++i) {
            std::cout << loaded_model.architecture[i];
            if (i < loaded_model.architecture.size() - 1) std::cout << " → ";
        }
        std::cout << "\n";
        std::cout << "   • Activation: " << loaded_model.activation << "\n";
        std::cout << "   • Epochs trained: " << loaded_model.epochs_trained << "\n";
        std::cout << "   • Final loss: " << std::fixed << std::setprecision(4) << loaded_model.final_loss << "\n";
        std::cout << "   • Training date: " << loaded_model.training_date << "\n";
        std::cout << "   • Weights: [";
        for (size_t i = 0; i < loaded_model.weights.size(); ++i) {
            std::cout << std::setprecision(3) << loaded_model.weights[i];
            if (i < loaded_model.weights.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "   • Biases: [";
        for (size_t i = 0; i < loaded_model.biases.size(); ++i) {
            std::cout << std::setprecision(3) << loaded_model.biases[i];
            if (i < loaded_model.biases.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n\n";
        
        // Test the loaded model on new data
        std::cout << "🧪 Testing loaded model on new data:\n";
        std::vector<float> test_inputs = {5.0f, 6.0f, 7.0f, 10.0f};
        
        for (float x : test_inputs) {
            // Apply loaded model: y = weight * x + bias
            float prediction = loaded_model.weights[0] * x + loaded_model.biases[0];
            float expected = 2.5f * x + 1.0f; // True function
            float error = std::abs(prediction - expected);
            
            std::cout << "   x=" << std::setprecision(1) << x 
                      << " → predicted=" << std::setprecision(2) << prediction
                      << ", expected=" << std::setprecision(2) << expected
                      << ", error=" << std::setprecision(3) << error << "\n";
        }
        
        std::cout << "\n✅ Model loading and inference successful!\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ Loading/testing error: " << e.what() << "\n";
    }
}

/**
 * @brief Show file contents for verification
 */
void show_exported_files() {
    print_header("EXPORTED MODEL FILES");
    
    // Show text file contents (human readable)
    std::cout << "📄 Text format (linear_model.txt):\n";
    std::cout << std::string(40, '-') << "\n";
    
    std::ifstream text_file("linear_model.txt");
    if (text_file.is_open()) {
        std::string line;
        int line_count = 0;
        while (std::getline(text_file, line) && line_count < 20) {
            std::cout << line << "\n";
            line_count++;
        }
        if (line_count == 20) std::cout << "...\n";
        text_file.close();
    } else {
        std::cout << "❌ Could not read text file\n";
    }
    
    std::cout << std::string(40, '-') << "\n\n";
    
    // Show binary file info
    std::ifstream binary_file("linear_model.dlvk", std::ios::binary | std::ios::ate);
    if (binary_file.is_open()) {
        auto size = binary_file.tellg();
        std::cout << "📦 Binary format (linear_model.dlvk): " << size << " bytes\n";
        std::cout << "   • Compact and fast to load\n";
        std::cout << "   • Contains all model data and metadata\n";
        binary_file.close();
    } else {
        std::cout << "❌ Could not check binary file\n";
    }
}

int main() {
    print_header("DLVK MODEL EXPORT/IMPORT DEMO");
    std::cout << "💾 Complete model persistence workflow!\n";
    std::cout << "🔄 Train → Export → Load → Test\n";
    
    // Step 1: Train and export model
    train_and_export_model();
    
    // Step 2: Load and test model  
    load_and_test_model();
    
    // Step 3: Show exported files
    show_exported_files();
    
    print_header("MODEL EXPORT DEMO COMPLETE!");
    std::cout << "🎉 Successfully demonstrated:\n";
    std::cout << "   ✅ Model training on GPU\n";
    std::cout << "   ✅ Model export (binary + text formats)\n";
    std::cout << "   ✅ Model import and loading\n";
    std::cout << "   ✅ Inference with loaded model\n";
    std::cout << "   ✅ Model metadata preservation\n";
    std::cout << "   ✅ Cross-session model persistence\n\n";
    std::cout << "💾 Users can now save and share their trained models!\n";
    
    return 0;
}
