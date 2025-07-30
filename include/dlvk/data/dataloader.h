#pragma once

#include "dataset.h"
#include <random>

namespace dlvk {

class VulkanDevice; // Forward declaration

namespace data {

/**
 * @brief Data loader for batching and shuffling datasets
 * Provides efficient batch processing for training
 */
class DataLoader {
private:
    std::shared_ptr<Dataset> m_dataset;
    std::shared_ptr<VulkanDevice> m_device;
    size_t m_batch_size;
    bool m_shuffle;
    bool m_drop_last;
    std::vector<size_t> m_indices;
    mutable std::mt19937 m_rng;

    void shuffle_indices();

public:
    /**
     * @brief Constructor for DataLoader
     * @param dataset The dataset to load from
     * @param device The Vulkan device for tensor creation
     * @param batch_size Number of samples per batch
     * @param shuffle Whether to shuffle data each epoch
     * @param drop_last Whether to drop the last incomplete batch
     */
    DataLoader(std::shared_ptr<Dataset> dataset,
              std::shared_ptr<VulkanDevice> device,
              size_t batch_size = 32,
              bool shuffle = true,
              bool drop_last = false);

    /**
     * @brief Get the number of batches
     */
    size_t num_batches() const;

    /**
     * @brief Get a batch of data
     * @param batch_idx The batch index to retrieve
     * @return A pair of (input_batch, target_batch) tensors
     */
    std::pair<Tensor, Tensor> get_batch(size_t batch_idx) const;

    /**
     * @brief Iterator class for DataLoader
     */
    class Iterator {
    private:
        const DataLoader* m_loader;
        size_t m_current_batch;

    public:
        Iterator(const DataLoader* loader, size_t batch_idx);
        
        std::pair<Tensor, Tensor> operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
    };

    /**
     * @brief Begin iterator for range-based for loops
     */
    Iterator begin() const;

    /**
     * @brief End iterator for range-based for loops
     */
    Iterator end() const;

    /**
     * @brief Shuffle data for new epoch
     */
    void new_epoch();


    size_t batch_size() const { return m_batch_size; }
    size_t dataset_size() const { return m_dataset->size(); }
};

} // namespace data
} // namespace dlvk
