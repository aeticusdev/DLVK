#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include "dlvk/core/vulkan_device.h"
#include "dlvk/core/vulkan_device.h"
#include "dlvk/tensor/tensor.h"
#include "dlvk/model/model.h"
#include "dlvk/training/trainer.h"
#include "dlvk/data/dataloader.h"
#include "dlvk/compute/compute_pipeline.h"

namespace dlvk {
namespace deployment {

/**
 * @brief Multi-GPU distributed training system
 * 
 * Provides data parallelism across multiple GPUs with gradient synchronization
 * and automatic load balancing for scalable training of large models.
 */
class MultiGPUTrainer {
public:
    struct Config {
        std::vector<int> device_ids;              // GPU device IDs to use
        size_t master_device = 0;                 // Master device for gradient aggregation
        bool synchronous_updates = true;         // Synchronous vs asynchronous gradient updates
        size_t gradient_accumulation_steps = 1;  // Steps before gradient sync
        bool use_nccl = false;                   // Use NCCL for communication (if available)
        float communication_overlap = 0.5f;      // Overlap computation with communication
    };

    training::TrainingMetrics train_replica(size_t replica_id, data::DataLoader& train_loader);

    struct DeviceReplica {
        std::shared_ptr<VulkanDevice> device;
        std::unique_ptr<Sequential> model;
        std::unique_ptr<training::Trainer> trainer;
        std::thread worker_thread;
        std::mutex gradient_mutex;
        std::vector<std::shared_ptr<Tensor>> gradients;
        bool ready_for_sync = false;
    };

private:
    Config config_;
    std::vector<std::unique_ptr<DeviceReplica>> replicas_;
    std::shared_ptr<VulkanDevice> master_device_;
    
    // Synchronization primitives
    std::mutex sync_mutex_;
    std::condition_variable sync_cv_;
    std::atomic<int> devices_ready_{0};
    std::atomic<bool> training_active_{false};
    
    // Communication buffers
    std::vector<std::shared_ptr<Tensor>> master_gradients_;
    std::vector<std::future<void>> communication_futures_;
    
    // Statistics
    struct Stats {
        std::atomic<size_t> total_batches{0};
        std::atomic<size_t> sync_operations{0};
        std::atomic<double> communication_time{0.0};
        std::atomic<double> computation_time{0.0};
    } stats_;

public:
    explicit MultiGPUTrainer(const Config& config);
    ~MultiGPUTrainer();

    /**
     * @brief Initialize multi-GPU training setup
     * @param base_model Model to replicate across devices
     * @return Success status
     */
    bool initialize(const Sequential& base_model);

    /**
     * @brief Train model across multiple GPUs
     * @param train_loader Training data loader
     * @param val_loader Validation data loader (optional)
     * @param epochs Number of training epochs
     * @param callbacks Training callbacks
     * @return Training metrics
     */
    training::TrainingMetrics train(
        data::DataLoader& train_loader,
        data::DataLoader* val_loader = nullptr,
        int epochs = 1,
        const std::vector<std::unique_ptr<training::TrainingCallback>>& callbacks = {}
    );

    /**
     * @brief Evaluate model performance across devices
     * @param data_loader Evaluation data loader
     * @return Evaluation metrics
     */
    training::TrainingMetrics evaluate(data::DataLoader& data_loader);

    /**
     * @brief Get consolidated model from master device
     * @return Master model with averaged weights
     */
    std::unique_ptr<Sequential> get_consolidated_model() const;

    /**
     * @brief Get training statistics
     * @return Performance statistics
     */
    struct TrainingStats {
        size_t total_batches;
        size_t sync_operations;
        double communication_time;
        double computation_time;
        double communication_efficiency;
        double scaling_efficiency;
    };

    TrainingStats get_stats() const;

    /**
     * @brief Save distributed training checkpoint
     * @param checkpoint_path Path to save checkpoint
     * @return Success status
     */
    bool save_checkpoint([[maybe_unused]]const std::string& checkpoint_path) const;

    /**
     * @brief Load distributed training checkpoint
     * @param checkpoint_path Path to load checkpoint
     * @return Success status
     */
    bool load_checkpoint([[maybe_unused]] const std::string& checkpoint_path);

private:
    void initialize_replica(int device_id, const Sequential& base_model);
    void worker_thread_func(int replica_id, data::DataLoader& train_loader);
    void synchronize_gradients();
    void all_reduce_gradients();
    void broadcast_parameters();
    void aggregate_metrics(std::vector<training::TrainingMetrics>& device_metrics);

    // Communication optimization
    void optimize_communication_schedule();
    void overlap_computation_communication();
    
    // Load balancing
    void balance_data_distribution(data::DataLoader& train_loader);
    std::vector<size_t> calculate_device_batch_sizes(size_t total_batch_size);
};

/**
 * @brief Utility class for multi-GPU device management
 */
class MultiGPUDeviceManager {
public:
    struct DeviceInfo {
        int device_id;
        std::string device_name;
        size_t memory_size;
        float compute_capability;
        bool is_available;
    };

    /**
     * @brief Detect available GPUs
     * @return List of available GPU devices
     */
    static std::vector<DeviceInfo> detect_gpus();

    /**
     * @brief Select optimal GPU configuration
     * @param available_gpus Available GPU devices
     * @param model_size Estimated model size in bytes
     * @param batch_size Training batch size
     * @return Optimal device configuration
     */
    static MultiGPUTrainer::Config select_optimal_config(
        const std::vector<DeviceInfo>& available_gpus,
        size_t model_size,
        size_t batch_size
    );

    /**
     * @brief Estimate memory requirements per device
     * @param model Model to analyze
     * @param batch_size Training batch size
     * @return Memory requirement in bytes
     */
    static size_t estimate_memory_requirements(
        const Sequential& model,
        size_t batch_size
    );

    /**
     * @brief Check GPU compatibility for distributed training
     * @param device_ids GPU device IDs to check
     * @return Compatibility status
     */
    static bool check_compatibility(const std::vector<int>& device_ids);
};

/**
 * @brief Gradient synchronization strategies
 */
enum class GradientSyncStrategy {
    AllReduce,          // Standard all-reduce operation
    ParameterServer,    // Parameter server architecture
    HierarchicalReduce, // Hierarchical reduction for large clusters
    AsyncUpdate        // Asynchronous parameter updates
};

/**
 * @brief Communication backend for multi-GPU training
 */
class CommunicationBackend {
public:
    virtual ~CommunicationBackend() = default;

    virtual bool initialize(const std::vector<int>& device_ids) = 0;
    virtual void all_reduce(std::vector<std::shared_ptr<Tensor>>& tensors) = 0;
    virtual void broadcast(std::shared_ptr<Tensor> tensor, int root_device) = 0;
    virtual void all_gather(const std::vector<std::shared_ptr<Tensor>>& input_tensors,
                           std::vector<std::shared_ptr<Tensor>>& output_tensors) = 0;
    virtual void reduce_scatter(const std::vector<std::shared_ptr<Tensor>>& input_tensors,
                               std::vector<std::shared_ptr<Tensor>>& output_tensors) = 0;
    virtual void finalize() = 0;
};

/**
 * @brief Vulkan-based communication backend
 */
class VulkanCommunicationBackend : public CommunicationBackend {
private:
    std::vector<std::shared_ptr<VulkanDevice>> devices_;
    std::vector<std::unique_ptr<ComputePipeline>> communication_pipelines_;

public:
    bool initialize(const std::vector<int>& device_ids) override;
    void all_reduce(std::vector<std::shared_ptr<Tensor>>& tensors) override;
    void broadcast(std::shared_ptr<Tensor> tensor, int root_device) override;
    void all_gather(const std::vector<std::shared_ptr<Tensor>>& input_tensors,
                   std::vector<std::shared_ptr<Tensor>>& output_tensors) override;
    void reduce_scatter(const std::vector<std::shared_ptr<Tensor>>& input_tensors,
                       std::vector<std::shared_ptr<Tensor>>& output_tensors) override;
    void finalize() override;

private:
    void create_communication_pipelines();
    void perform_ring_all_reduce(std::vector<std::shared_ptr<Tensor>>& tensors);
    void perform_tree_reduce(std::vector<std::shared_ptr<Tensor>>& tensors);
};

} // namespace deployment
} // namespace dlvk
