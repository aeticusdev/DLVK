#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>

#include "dlvk/tensor/tensor.h"
#include "dlvk/model/model.h"
#include "dlvk/core/vulkan_device.h"

namespace dlvk {
namespace optimization {

/**
 * @brief Quantization data types supported
 */
enum class QuantizationType {
    INT8,    // 8-bit integer quantization
    INT16,   // 16-bit integer quantization
    FP16,    // 16-bit floating point
    DYNAMIC, // Dynamic quantization
    STATIC   // Static quantization with calibration
};

/**
 * @brief Pruning strategies
 */
enum class PruningStrategy {
    MAGNITUDE,     // Magnitude-based pruning
    STRUCTURED,    // Structured pruning (entire neurons/channels)
    GRADUAL,       // Gradual pruning during training
    LOTTERY_TICKET // Lottery ticket hypothesis
};

/**
 * @brief Quantization configuration
 */
struct QuantizationConfig {
    QuantizationType type = QuantizationType::INT8;
    bool symmetric = true;
    bool per_channel = true;
    std::vector<std::string> quantized_layers;  // Layers to quantize (empty = all)
    std::vector<std::string> skip_layers;       // Layers to skip
    float calibration_data_ratio = 0.1f;        // Fraction of data for calibration
};

/**
 * @brief Pruning configuration
 */
struct PruningConfig {
    PruningStrategy strategy = PruningStrategy::MAGNITUDE;
    float sparsity_ratio = 0.5f;               // Target sparsity (0.0 to 1.0)
    std::vector<std::string> pruned_layers;    // Layers to prune (empty = all)
    std::vector<std::string> skip_layers;       // Layers to skip
    bool gradual_pruning = false;
    int pruning_frequency = 100;               // Steps between pruning updates
};

/**
 * @brief ONNX export configuration
 */
struct ONNXConfig {
    int opset_version = 11;
    bool optimize_for_inference = true;
    bool include_initializers = true;
    std::map<std::string, std::vector<int64_t>> dynamic_axes;
    std::string model_name = "dlvk_model";
    std::string model_version = "1.0";
};

/**
 * @brief Model optimization statistics
 */
struct OptimizationStats {
    size_t original_parameters = 0;
    size_t optimized_parameters = 0;
    size_t original_model_size = 0;     // in bytes
    size_t optimized_model_size = 0;    // in bytes
    float compression_ratio = 1.0f;
    float accuracy_drop = 0.0f;         // Accuracy difference after optimization
    double optimization_time = 0.0;     // Time taken for optimization
};

/**
 * @brief Model optimizer for production deployment
 */
class ModelOptimizer {
public:
    /**
     * @brief Quantize model to reduced precision
     * @param model Model to be quantized
     * @param config Quantization configuration
     * @param calibration_data Data for calibration (required for static quantization)
     * @return Quantized model and optimization statistics
     */
    static std::pair<std::unique_ptr<Sequential>, OptimizationStats> quantize_model(
        const Sequential& model,
        const QuantizationConfig& config = {},
        const std::vector<std::shared_ptr<Tensor>>& calibration_data = {}
    );

    /**
     * @brief Prune model to remove less important weights
     * @param model Model to be pruned
     * @param config Pruning configuration
     * @return Pruned model and optimization statistics
     */
    static std::pair<std::unique_ptr<Sequential>, OptimizationStats> prune_model(
        const Sequential& model,
        const PruningConfig& config = {}
    );

    /**
     * @brief Apply knowledge distillation for model compression
     * @param teacher_model Large teacher model
     * @param student_model Smaller student model
     * @param temperature Distillation temperature
     * @param alpha Weight for distillation loss
     * @return Compressed student model
     */
    static std::unique_ptr<Sequential> distill_model(
        const Sequential& teacher_model,
        Sequential& student_model,
        float temperature = 4.0f,
        float alpha = 0.7f
    );

    /**
     * @brief Export model to ONNX format
     * @param model Model to be exported
     * @param file_path File path to save ONNX model
     * @param config ONNX export configuration
     * @return Success status
     */
    static bool export_to_onnx(
        const Sequential& model,
        const std::string& file_path,
        const ONNXConfig& config = {}
    );

    /**
     * @brief Import model from ONNX format
     * @param file_path File path to the ONNX model
     * @param device Vulkan device for model creation
     * @return Imported model
     */
    static std::unique_ptr<Sequential> import_from_onnx(
        const std::string& file_path,
        std::shared_ptr<VulkanDevice> device
    );

    /**
     * @brief Optimize model for inference
     * @param model Model to optimize
     * @param target_device Target deployment device
     * @return Optimized model for inference
     */
    static std::unique_ptr<Sequential> optimize_for_inference(
        const Sequential& model,
        const std::string& target_device = "cpu"
    );

    /**
     * @brief Benchmark model performance
     * @param model Model to benchmark
     * @param input_shape Input tensor shape
     * @param batch_sizes Batch sizes to test
     * @param num_runs Number of benchmark runs
     * @return Performance statistics
     */
    struct BenchmarkResults {
        std::map<size_t, double> latency_per_batch_size;  // ms
        std::map<size_t, double> throughput_per_batch_size; // samples/sec
        size_t peak_memory_usage;                         // bytes
        double average_gpu_utilization;                   // percentage
    };

    static BenchmarkResults benchmark_model(
        const Sequential& model,
        const std::vector<size_t>& input_shape,
        const std::vector<size_t>& batch_sizes = {1, 8, 16, 32},
        int num_runs = 100
    );

private:

    static void quantize_weights(std::shared_ptr<Tensor> weights, const QuantizationConfig& config);
    static void calibrate_quantization_scales(
        const Sequential& model,
        const std::vector<std::shared_ptr<Tensor>>& calibration_data,
        std::map<std::string, float>& scales
    );


    static void apply_magnitude_pruning(std::shared_ptr<Tensor> weights, float threshold);
    static void apply_structured_pruning(Sequential& model, const PruningConfig& config);
    static std::vector<int> select_neurons_to_prune(
        const std::shared_ptr<Tensor> weights,
        float sparsity_ratio
    );


    static void build_onnx_graph(const Sequential& model, const ONNXConfig& config);
    static void convert_layer_to_onnx_node(const ModernLayer& layer);
    static Sequential parse_onnx_graph(const std::string& file_path, std::shared_ptr<VulkanDevice> device);


    static OptimizationStats calculate_optimization_stats(
        const Sequential& original_model,
        const Sequential& optimized_model
    );
    static size_t estimate_model_size(const Sequential& model);
    static size_t count_parameters(const Sequential& model);
};

/**
 * @brief Neural Architecture Search (NAS) for automatic model optimization
 */
class NeuralArchitectureSearch {
public:
    struct SearchSpace {
        std::vector<size_t> layer_widths;
        std::vector<size_t> layer_depths;
        std::vector<std::string> activation_functions;
        std::vector<float> dropout_rates;
    };

    struct SearchConfig {
        SearchSpace search_space;
        size_t population_size = 20;
        size_t num_generations = 50;
        float mutation_rate = 0.1f;
        float crossover_rate = 0.7f;
        std::function<float(const Sequential&)> fitness_function;
    };

    /**
     * @brief Search for optimal model architecture
     * @param config Search configuration
     * @param device Vulkan device
     * @return Optimal model architecture
     */
    static std::unique_ptr<Sequential> search_architecture(
        const SearchConfig& config,
        std::shared_ptr<VulkanDevice> device
    );

private:
    static std::unique_ptr<Sequential> generate_random_architecture(
        const SearchSpace& space,
        std::shared_ptr<VulkanDevice> device
    );
    static std::unique_ptr<Sequential> mutate_architecture(
        const Sequential& parent,
        const SearchSpace& space,
        float mutation_rate
    );
    static std::unique_ptr<Sequential> crossover_architectures(
        const Sequential& parent1,
        const Sequential& parent2,
        const SearchSpace& space
    );
};

} // namespace optimization
} // namespace dlvk

