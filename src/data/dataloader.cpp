#include "dlvk/data/dataloader.h"
#include "dlvk/data/mnist.h"
#include "dlvk/core/vulkan_device.h"
#include <algorithm>
#include <random>

namespace dlvk {
namespace data {

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset,
                      std::shared_ptr<VulkanDevice> device,
                      size_t batch_size,
                      bool shuffle,
                      bool drop_last)
    : m_dataset(std::move(dataset))
    , m_device(device)
    , m_batch_size(batch_size)
    , m_shuffle(shuffle)
    , m_drop_last(drop_last)
    , m_rng(std::random_device{}()) {
    
    // Initialize indices
    m_indices.resize(m_dataset->size());
    std::iota(m_indices.begin(), m_indices.end(), 0);
    
    if (m_shuffle) {
        shuffle_indices();
    }
}

void DataLoader::shuffle_indices() {
    std::shuffle(m_indices.begin(), m_indices.end(), m_rng);
}

size_t DataLoader::num_batches() const {
    if (m_drop_last) {
        return m_dataset->size() / m_batch_size;
    } else {
        return (m_dataset->size() + m_batch_size - 1) / m_batch_size;
    }
}

std::pair<Tensor, Tensor> DataLoader::get_batch(size_t batch_idx) const {
    if (batch_idx >= num_batches()) {
        throw std::out_of_range("Batch index out of range");
    }

    size_t start_idx = batch_idx * m_batch_size;
    size_t end_idx = std::min(start_idx + m_batch_size, m_dataset->size());
    size_t actual_batch_size = end_idx - start_idx;

    // Special handling for MNIST dataset
    auto mnist_dataset = std::dynamic_pointer_cast<MnistDataset>(m_dataset);
    if (mnist_dataset) {
        // Get shapes from the dataset
        auto input_shape = mnist_dataset->input_shape();
        auto target_shape = mnist_dataset->target_shape();

        // Create batch shapes (add batch dimension)
        std::vector<size_t> batch_input_shape = {actual_batch_size};
        batch_input_shape.insert(batch_input_shape.end(), input_shape.begin(), input_shape.end());
        
        std::vector<size_t> batch_target_shape = {actual_batch_size};
        batch_target_shape.insert(batch_target_shape.end(), target_shape.begin(), target_shape.end());

        // Calculate total sizes
        size_t input_sample_size = 1;
        for (size_t dim : input_shape) {
            input_sample_size *= dim;
        }
        
        size_t target_sample_size = 1;
        for (size_t dim : target_shape) {
            target_sample_size *= dim;
        }

        // Allocate batch data
        std::vector<float> batch_input_data(actual_batch_size * input_sample_size);
        std::vector<float> batch_target_data(actual_batch_size * target_sample_size);

        // Fill batch data
        for (size_t i = 0; i < actual_batch_size; ++i) {
            size_t sample_idx = m_indices[start_idx + i];
            
            // Copy input data (image)
            const auto& image_data = mnist_dataset->get_image_data(sample_idx);
            std::copy(image_data.begin(), image_data.end(),
                     batch_input_data.begin() + i * input_sample_size);
            
            // Create one-hot encoded target
            int label = mnist_dataset->get_label(sample_idx);
            size_t target_offset = i * target_sample_size;
            std::fill(batch_target_data.begin() + target_offset,
                     batch_target_data.begin() + target_offset + target_sample_size, 0.0f);
            batch_target_data[target_offset + label] = 1.0f;
        }

        // Create batch tensors
        Tensor batch_input(batch_input_shape, DataType::FLOAT32, m_device);
        Tensor batch_target(batch_target_shape, DataType::FLOAT32, m_device);
        
        // Upload data to GPU
        batch_input.upload_data(batch_input_data.data());
        batch_target.upload_data(batch_target_data.data());

        return {std::move(batch_input), std::move(batch_target)};
    }

    // Fallback for other dataset types (not implemented yet)
    throw std::runtime_error("DataLoader only supports MnistDataset currently");
}

void DataLoader::new_epoch() {
    if (m_shuffle) {
        shuffle_indices();
    }
}

// Iterator implementation
DataLoader::Iterator::Iterator(const DataLoader* loader, size_t batch_idx)
    : m_loader(loader), m_current_batch(batch_idx) {
}

std::pair<Tensor, Tensor> DataLoader::Iterator::operator*() const {
    return m_loader->get_batch(m_current_batch);
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    ++m_current_batch;
    return *this;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return m_current_batch != other.m_current_batch;
}

DataLoader::Iterator DataLoader::begin() const {
    return Iterator(this, 0);
}

DataLoader::Iterator DataLoader::end() const {
    return Iterator(this, num_batches());
}

} // namespace data
} // namespace dlvk
