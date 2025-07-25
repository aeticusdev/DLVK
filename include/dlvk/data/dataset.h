#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "dlvk/tensor/tensor.h"

namespace dlvk {
namespace data {

/**
 * @brief Abstract base class for all datasets
 * Provides a unified interface for different data sources
 */
class Dataset {
public:
    virtual ~Dataset() = default;

    /**
     * @brief Get the number of samples in the dataset
     */
    virtual size_t size() const = 0;

    /**
     * @brief Get a single sample from the dataset
     * @param index The index of the sample to retrieve
     * @return A pair of (input, target) tensors
     */
    virtual std::pair<Tensor, Tensor> get_item(size_t index) const = 0;

    /**
     * @brief Get the shape of input samples
     */
    virtual std::vector<size_t> input_shape() const = 0;

    /**
     * @brief Get the shape of target samples
     */
    virtual std::vector<size_t> target_shape() const = 0;

    /**
     * @brief Get the number of classes (for classification datasets)
     */
    virtual size_t num_classes() const = 0;
};

/**
 * @brief Data transformation function type
 * Takes a tensor and returns a transformed tensor
 */
using Transform = std::function<Tensor(const Tensor&)>;

/**
 * @brief Dataset wrapper that applies transformations
 */
class TransformDataset : public Dataset {
private:
    std::shared_ptr<Dataset> m_base_dataset;
    Transform m_input_transform;
    Transform m_target_transform;

public:
    TransformDataset(std::shared_ptr<Dataset> dataset,
                    Transform input_transform = nullptr,
                    Transform target_transform = nullptr);

    size_t size() const override;
    std::pair<Tensor, Tensor> get_item(size_t index) const override;
    std::vector<size_t> input_shape() const override;
    std::vector<size_t> target_shape() const override;
    size_t num_classes() const override;
};

} // namespace data
} // namespace dlvk
