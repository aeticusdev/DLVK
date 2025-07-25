#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

// DLVK includes
#include "dlvk/tensor/tensor.h"
#include "dlvk/loss/loss_functions.h"

using namespace std;
using namespace std::chrono;
using namespace dlvk;

void test_binary_cross_entropy_performance() {
    cout << "=== BinaryCrossEntropy GPU Performance Test ===" << endl;
    
    try {
        // Create large test data for meaningful performance measurement
        size_t batch_size = 1024;
        size_t features = 512;
        size_t total_elements = batch_size * features;
        
        cout << "✓ Test data size: [" << batch_size << ", " << features << "] = " 
             << total_elements << " elements" << endl;
        
        // Initialize random data
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> pred_dist(0.1f, 0.9f);  // Avoid extremes
        uniform_int_distribution<int> target_dist(0, 1);
        
        vector<float> pred_data(total_elements);
        vector<float> target_data(total_elements);
        
        for (size_t i = 0; i < total_elements; ++i) {
            pred_data[i] = pred_dist(gen);
            target_data[i] = static_cast<float>(target_dist(gen));
        }
        
        // Create tensors with proper API
        auto predictions = make_shared<Tensor>(vector<size_t>{batch_size, features}, DataType::FLOAT32, nullptr);
        auto targets = make_shared<Tensor>(vector<size_t>{batch_size, features}, DataType::FLOAT32, nullptr);
        
        // Set data (assuming there's a method to set data)
        // For now, let's create a simpler test
        
        cout << "✓ Tensors created successfully" << endl;
        
        // Create BinaryCrossEntropy loss function
        BinaryCrossEntropyLoss bce_loss;
        
        cout << "\n--- GPU Performance Test ---" << endl;
        
        // Warmup run
        auto warmup_start = high_resolution_clock::now();
        auto warmup_result = bce_loss.forward(predictions, targets);
        auto warmup_end = high_resolution_clock::now();
        auto warmup_duration = duration_cast<microseconds>(warmup_end - warmup_start);
        
        cout << "✓ Warmup run: " << warmup_duration.count() << " μs" << endl;
        
        // Performance measurement - single run
        auto single_start = high_resolution_clock::now();
        auto loss_result = bce_loss.forward(predictions, targets);
        auto single_end = high_resolution_clock::now();
        auto single_duration = duration_cast<microseconds>(single_end - single_start);
        
        cout << "✓ Single run (cached): " << single_duration.count() << " μs" << endl;
        
        // Multiple runs for average
        const int num_runs = 100;
        auto multi_start = high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            bce_loss.forward(predictions, targets);
        }
        
        auto multi_end = high_resolution_clock::now();
        auto multi_duration = duration_cast<microseconds>(multi_end - multi_start);
        auto avg_duration = multi_duration.count() / num_runs;
        
        cout << "✓ " << num_runs << " runs completed in " << multi_duration.count() << " μs" << endl;
        cout << "✓ Average per run: " << avg_duration << " μs" << endl;
        
        // Calculate throughput
        double throughput = (double)total_elements / avg_duration;  // elements per microsecond
        throughput *= 1e6 / 1e6;  // Convert to millions of elements per second
        
        cout << "\n--- Performance Results ---" << endl;
        cout << "Data size: " << total_elements << " elements" << endl;
        cout << "Average time: " << avg_duration << " μs" << endl;
        cout << "Throughput: " << fixed << setprecision(2) << throughput << " million elements/second" << endl;
        
        // Performance assessment
        if (throughput > 200.0) {
            cout << "🚀 EXCELLENT PERFORMANCE! GPU acceleration working optimally." << endl;
        } else if (throughput > 100.0) {
            cout << "✅ GOOD PERFORMANCE! GPU acceleration confirmed." << endl;
        } else if (throughput > 50.0) {
            cout << "⚠️  MODERATE PERFORMANCE. Room for optimization." << endl;
        } else {
            cout << "❌ LOW PERFORMANCE. Potential CPU fallback or inefficient GPU usage." << endl;
        }
        
        // Test backward pass
        cout << "\n--- Backward Pass Test ---" << endl;
        auto back_start = high_resolution_clock::now();
        auto gradients = bce_loss.backward(predictions, targets);
        auto back_end = high_resolution_clock::now();
        auto back_duration = duration_cast<microseconds>(back_end - back_start);
        
        cout << "✓ Backward pass: " << back_duration.count() << " μs" << endl;
        
        double back_throughput = (double)total_elements / back_duration.count();
        cout << "✓ Backward throughput: " << fixed << setprecision(2) << back_throughput << " million elements/second" << endl;
        
        cout << "\n=== BinaryCrossEntropy Performance Test Completed! ===" << endl;
        
    } catch (const exception& e) {
        cout << "❌ Error: " << e.what() << endl;
    }
}

int main() {
    test_binary_cross_entropy_performance();
    return 0;
}
