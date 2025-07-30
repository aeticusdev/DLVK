#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <functional>
#include "dlvk/model/model.h"
#include "dlvk/training/trainer.h"

namespace dlvk {

/**
 * @brief Serialization formats supported by DLVK
 */
enum class SerializationFormat {
    DLVK_BINARY,
    DLVK_JSON,
    HDF5,
    ONNX,
    NUMPY_NPZ
};

/**
 * @brief Model metadata for persistence
 */
struct ModelMetadata {
    std::string name;
    std::string description;
    std::string version;
    std::string framework_version;
    std::unordered_map<std::string, std::string> custom_fields;
    

    std::string model_name;  // alias for name
};

/**
 * @brief Checkpoint data structure
 */
struct CheckpointData {
    ModelMetadata metadata;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;
    

    std::vector<std::vector<float>> optimizer_state;
    std::unordered_map<std::string, float> optimizer_params;
    

    int current_epoch = 0;
    int current_step = 0;
    float current_lr = 0.0f;
    dlvk::TrainingMetrics current_metrics;
    std::vector<dlvk::TrainingMetrics> training_history;
    

    uint32_t random_seed = 0;
    std::vector<uint32_t> rng_state;
};

namespace training {

/**
 * @brief Model checkpoint manager
 */
class ModelCheckpoint {
private:
    std::string m_checkpoint_dir;
    std::string m_filename_prefix; // Moved up
    int m_save_frequency;         // Moved up
    std::string m_model_name;
    SerializationFormat m_format;
    bool m_save_best_only;
    std::string m_monitor_metric;
    bool m_maximize_metric;
    float m_best_metric_value;
    std::string m_best_checkpoint_path;
    int m_max_checkpoints;
    int m_checkpoint_count;
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


    ModelCheckpoint(const std::string& directory, 
                   const std::string& prefix,
                   int save_frequency);
    
    /**
     * @brief Save model checkpoint
     */
    bool save_checkpoint(const std::shared_ptr<dlvk::Model>& model,
                        const dlvk::training::TrainingMetrics& metrics,
                        int epoch,
                        std::shared_ptr<dlvk::Optimizer> optimizer = nullptr);

    bool save_checkpoint(const std::shared_ptr<dlvk::Model>& model,
                        const dlvk::training::TrainingMetrics& metrics,
                        int epoch);

    /**
     * @brief Load model checkpoint
     */
    bool load_checkpoint(std::shared_ptr<Model>& model,
                        const std::string& checkpoint_path,
                        std::shared_ptr<Optimizer> optimizer = nullptr);


    bool load_checkpoint(const std::string& filepath,
                        std::shared_ptr<Model>& model,
                        TrainingMetrics& metrics);

    /**
     * @brief Save metrics to file
     */
    bool save_metrics(const TrainingMetrics& metrics,
                     const std::string& filepath,
                     int epoch);
    
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
    ModelSerializer() = default;
    ~ModelSerializer() = default;

    /**
     * @brief Save model in binary format
     */
    bool save_binary(const std::shared_ptr<Model>& model,
                    const std::string& filepath,
                    const ::dlvk::ModelMetadata& metadata = {});

    /**
     * @brief Save model in JSON format
     */
    bool save_json(const std::shared_ptr<Model>& model,
                  const std::string& filepath,
                  const ::dlvk::ModelMetadata& metadata = {});

    /**
     * @brief Load model from binary format
     */
    std::shared_ptr<Model> load_binary(const std::string& filepath);

    /**
     * @brief Load model from JSON format
     */
    std::shared_ptr<Model> load_json(const std::string& filepath);

    /**
     * @brief Serialize metadata to string
     */
    std::string serialize_metadata(const ::dlvk::ModelMetadata& metadata);

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

    static bool save_binary_format(const CheckpointData& data, const std::string& filepath);
    static bool load_binary_format(CheckpointData& data, const std::string& filepath);
    
    static bool save_json_format(const CheckpointData& data, const std::string& filepath);
    static bool load_json_format(CheckpointData& data, const std::string& filepath);
    

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
    

    std::string m_base_path;
    int m_current_version;
    
public:
    ModelVersioning(const std::string& experiment_dir) 
        : m_experiment_dir(experiment_dir), m_base_path(experiment_dir), m_current_version(1) {}
    
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
     * @brief Create version (implementation method)
     */
    std::string create_version(const std::shared_ptr<Model>& model,
                              const std::string& name,
                              const std::string& description);

    /**
     * @brief Save version info (implementation method)
     */
    bool save_version_info(const std::string& name,
                          const std::string& description,
                          const std::string& model_path);
    
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
class ModelCheckpointCallback : public training::TrainingCallback {
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
    
    void on_epoch_end(int epoch, const training::TrainingMetrics& metrics) override;
    
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
