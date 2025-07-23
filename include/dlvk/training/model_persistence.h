#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include "dlvk/model/model.h"
#include "dlvk/training/trainer.h"

namespace dlvk {
namespace training {

/**
 * @brief Model serialization formats
 */
enum class SerializationFormat {
    DLVK_BINARY,    // Custom binary format
    DLVK_JSON,      // JSON format with base64 weights
    HDF5,           // HDF5 format (requires H5 library)
    ONNX,           // ONNX format export
    NUMPY_NPZ       // NumPy .npz format
};

/**
 * @brief Model metadata for serialization
 */
struct ModelMetadata {
    std::string model_name;
    std::string model_version;
    std::string framework_version = "DLVK-0.1.0";
    std::string creation_date;
    std::string description;
    std::unordered_map<std::string, std::string> custom_metadata;
    
    // Architecture information
    std::vector<std::string> layer_types;
    std::vector<std::vector<int>> layer_shapes;
    int total_parameters = 0;
    int trainable_parameters = 0;
    
    // Training information
    int epochs_trained = 0;
    float final_loss = 0.0f;
    float final_accuracy = 0.0f;
    std::string optimizer_type;
    float learning_rate = 0.0f;
};

/**
 * @brief Checkpoint data structure
 */
struct CheckpointData {
    ModelMetadata metadata;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;
    
    // Optimizer state
    std::vector<std::vector<float>> optimizer_state;
    std::unordered_map<std::string, float> optimizer_params;
    
    // Training state
    int current_epoch = 0;
    int current_step = 0;
    float current_lr = 0.0f;
    TrainingMetrics current_metrics;
    std::vector<TrainingMetrics> training_history;
    
    // Random state for reproducibility
    uint32_t random_seed = 0;
    std::vector<uint32_t> rng_state;
};

/**
 * @brief Model checkpoint manager
 */
class ModelCheckpoint {
private:
    std::string m_checkpoint_dir;
    std::string m_model_name;
    SerializationFormat m_format;
    bool m_save_best_only;
    std::string m_monitor_metric;
    bool m_maximize_metric;
    
    float m_best_metric_value;
    std::string m_best_checkpoint_path;
    int m_save_frequency;  // Save every N epochs
    int m_max_checkpoints; // Keep only N latest checkpoints
    
    std::vector<std::string> m_checkpoint_history;
    
    void cleanup_old_checkpoints();
    std::string generate_checkpoint_filename(int epoch, float metric_value = 0.0f) const;
    
public:
    ModelCheckpoint(const std::string& checkpoint_dir,
                   const std::string& model_name,
                   SerializationFormat format = SerializationFormat::DLVK_BINARY,
                   bool save_best_only = true,
                   const std::string& monitor_metric = "val_loss",
                   bool maximize_metric = false,
                   int save_frequency = 1,
                   int max_checkpoints = 5)
        : m_checkpoint_dir(checkpoint_dir), m_model_name(model_name),
          m_format(format), m_save_best_only(save_best_only),
          m_monitor_metric(monitor_metric), m_maximize_metric(maximize_metric),
          m_save_frequency(save_frequency), m_max_checkpoints(max_checkpoints) {
        
        m_best_metric_value = maximize_metric ? 
            -std::numeric_limits<float>::infinity() : 
            std::numeric_limits<float>::infinity();
    }
    
    /**
     * @brief Save model checkpoint
     */
    bool save_checkpoint(const std::shared_ptr<Model>& model,
                        const TrainingMetrics& metrics,
                        int epoch,
                        const std::shared_ptr<Optimizer>& optimizer = nullptr);
    
    /**
     * @brief Load model checkpoint
     */
    bool load_checkpoint(std::shared_ptr<Model>& model,
                        const std::string& checkpoint_path,
                        std::shared_ptr<Optimizer>& optimizer = nullptr);
    
    /**
     * @brief Get best checkpoint path
     */
    const std::string& get_best_checkpoint_path() const { return m_best_checkpoint_path; }
    
    /**
     * @brief Get all checkpoint paths
     */
    const std::vector<std::string>& get_checkpoint_history() const { return m_checkpoint_history; }
    
    /**
     * @brief List available checkpoints in directory
     */
    std::vector<std::string> list_checkpoints() const;
    
    /**
     * @brief Set checkpoint directory
     */
    void set_checkpoint_directory(const std::string& dir) { m_checkpoint_dir = dir; }
};

/**
 * @brief Model serializer for different formats
 */
class ModelSerializer {
public:
    /**
     * @brief Save model to file
     */
    static bool save_model(const std::shared_ptr<Model>& model,
                          const std::string& filepath,
                          SerializationFormat format = SerializationFormat::DLVK_BINARY,
                          const ModelMetadata& metadata = {});
    
    /**
     * @brief Load model from file
     */
    static std::shared_ptr<Model> load_model(const std::string& filepath,
                                           SerializationFormat format = SerializationFormat::DLVK_BINARY);
    
    /**
     * @brief Save only model weights
     */
    static bool save_weights(const std::shared_ptr<Model>& model,
                            const std::string& filepath,
                            SerializationFormat format = SerializationFormat::DLVK_BINARY);
    
    /**
     * @brief Load only model weights
     */
    static bool load_weights(std::shared_ptr<Model>& model,
                            const std::string& filepath,
                            SerializationFormat format = SerializationFormat::DLVK_BINARY);
    
    /**
     * @brief Export model to ONNX format
     */
    static bool export_to_onnx(const std::shared_ptr<Model>& model,
                               const std::string& filepath,
                               const std::vector<std::vector<int>>& input_shapes);
    
    /**
     * @brief Export weights to NumPy format
     */
    static bool export_to_numpy(const std::shared_ptr<Model>& model,
                                const std::string& filepath);

private:
    // Format-specific implementations
    static bool save_binary_format(const CheckpointData& data, const std::string& filepath);
    static bool load_binary_format(CheckpointData& data, const std::string& filepath);
    
    static bool save_json_format(const CheckpointData& data, const std::string& filepath);
    static bool load_json_format(CheckpointData& data, const std::string& filepath);
    
    // Helper functions
    static CheckpointData model_to_checkpoint_data(const std::shared_ptr<Model>& model,
                                                  const ModelMetadata& metadata);
    static std::shared_ptr<Model> checkpoint_data_to_model(const CheckpointData& data);
    static std::string encode_base64(const std::vector<float>& data);
    static std::vector<float> decode_base64(const std::string& encoded);
};

/**
 * @brief Model versioning and experiment tracking
 */
class ModelVersioning {
private:
    std::string m_experiment_dir;
    std::string m_current_experiment;
    std::unordered_map<std::string, std::string> m_experiment_metadata;
    
public:
    ModelVersioning(const std::string& experiment_dir) 
        : m_experiment_dir(experiment_dir) {}
    
    /**
     * @brief Create new experiment
     */
    void create_experiment(const std::string& experiment_name,
                          const std::unordered_map<std::string, std::string>& metadata = {});
    
    /**
     * @brief Save model with version
     */
    std::string save_model_version(const std::shared_ptr<Model>& model,
                                  const std::string& version_tag,
                                  const ModelMetadata& metadata = {});
    
    /**
     * @brief Load specific model version
     */
    std::shared_ptr<Model> load_model_version(const std::string& version_tag);
    
    /**
     * @brief List all versions
     */
    std::vector<std::string> list_versions() const;
    
    /**
     * @brief Get experiment directory
     */
    std::string get_experiment_path() const;
    
    /**
     * @brief Compare model versions
     */
    struct VersionComparison {
        std::string version_a;
        std::string version_b;
        std::unordered_map<std::string, float> metrics_a;
        std::unordered_map<std::string, float> metrics_b;
        std::string better_version;
        std::string comparison_reason;
    };
    VersionComparison compare_versions(const std::string& version_a, 
                                     const std::string& version_b,
                                     const std::string& metric = "val_loss");
};

/**
 * @brief Model checkpoint callback for automatic saving during training
 */
class ModelCheckpointCallback : public TrainingCallback {
private:
    std::unique_ptr<ModelCheckpoint> m_checkpoint_manager;
    std::shared_ptr<Model> m_model;
    std::shared_ptr<Optimizer> m_optimizer;
    
public:
    ModelCheckpointCallback(std::shared_ptr<Model> model,
                           const std::string& checkpoint_dir,
                           const std::string& model_name,
                           std::shared_ptr<Optimizer> optimizer = nullptr,
                           bool save_best_only = true,
                           const std::string& monitor_metric = "val_loss")
        : m_model(model), m_optimizer(optimizer) {
        m_checkpoint_manager = std::make_unique<ModelCheckpoint>(
            checkpoint_dir, model_name, SerializationFormat::DLVK_BINARY,
            save_best_only, monitor_metric);
    }
    
    void on_epoch_end(int epoch, const TrainingMetrics& metrics) override;
    
    /**
     * @brief Get checkpoint manager
     */
    ModelCheckpoint& get_checkpoint_manager() { return *m_checkpoint_manager; }
};

/**
 * @brief Factory functions for common persistence setups
 */
namespace persistence_factory {
    
    /**
     * @brief Create standard model checkpoint callback
     */
    std::unique_ptr<ModelCheckpointCallback> create_checkpoint_callback(
        std::shared_ptr<Model> model,
        const std::string& checkpoint_dir = "./checkpoints",
        const std::string& model_name = "model",
        std::shared_ptr<Optimizer> optimizer = nullptr
    );
    
    /**
     * @brief Create model versioning system
     */
    std::unique_ptr<ModelVersioning> create_versioning_system(
        const std::string& experiment_dir = "./experiments"
    );
}

} // namespace training
} // namespace dlvk
