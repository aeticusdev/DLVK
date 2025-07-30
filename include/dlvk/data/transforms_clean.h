#pragma once

#include "dlvk/tensor/tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <random>
#include <unordered_map>

namespace dlvk {
namespace data {
namespace transforms {


class Transform;
class TensorTransform;
class AugmentationStrategy;
class PreprocessingPipeline;

/**
 * @brief Base class for all data transformations
 */
class Transform {
public:
    virtual ~Transform() = default;
    virtual Tensor operator()(const Tensor& input) const = 0;
    virtual std::unique_ptr<Transform> clone() const = 0;
};

/**
 * @brief Tensor-integrated transformation interface
 */
class TensorTransform : public Transform {
public:
    virtual ~TensorTransform() = default;
    virtual std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const = 0;
    Tensor operator()(const Tensor& input) const override;
    virtual std::unique_ptr<Transform> clone() const = 0;
};

/**
 * @brief Normalize tensor values to specified mean and std
 */
class Normalize : public TensorTransform {
private:
    std::vector<float> m_mean;
    std::vector<float> m_std;

public:
    Normalize(const std::vector<float>& mean, const std::vector<float>& std);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random horizontal flip
 */
class RandomHorizontalFlip : public TensorTransform {
private:
    float m_probability;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_dist;

public:
    explicit RandomHorizontalFlip(float probability = 0.5f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random crop with optional padding
 */
class RandomCrop : public TensorTransform {
private:
    size_t m_crop_height;
    size_t m_crop_width;
    size_t m_padding;
    mutable std::mt19937 m_rng;

public:
    RandomCrop(size_t crop_height, size_t crop_width, size_t padding = 0);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Resize image to specified dimensions
 */
class Resize : public TensorTransform {
public:
    enum class InterpolationMode { NEAREST, BILINEAR };

private:
    size_t m_height;
    size_t m_width;
    InterpolationMode m_mode;

public:
    Resize(size_t height, size_t width, InterpolationMode mode = InterpolationMode::BILINEAR);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Compose multiple transformations
 */
class Compose : public TensorTransform {
private:
    std::vector<std::shared_ptr<TensorTransform>> m_transforms;

public:
    explicit Compose(std::vector<std::shared_ptr<TensorTransform>> transforms);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
    
    void add_transform(std::shared_ptr<TensorTransform> transform);
    size_t size() const { return m_transforms.size(); }
};

/**
 * @brief Augmentation strategy base class
 */
class AugmentationStrategy {
public:
    virtual ~AugmentationStrategy() = default;
    virtual std::shared_ptr<Compose> create_transform() const = 0;
    virtual std::string name() const = 0;
};

/**
 * @brief Light augmentation strategy
 */
class LightAugmentation : public AugmentationStrategy {
public:
    std::shared_ptr<Compose> create_transform() const override;
    std::string name() const override { return "Light"; }
};

/**
 * @brief Custom preprocessing pipeline
 */
class PreprocessingPipeline {
private:
    std::shared_ptr<Compose> m_transforms;
    std::unordered_map<std::string, std::shared_ptr<TensorTransform>> m_named_transforms;

public:
    PreprocessingPipeline();
    

    PreprocessingPipeline& add_transform(const std::string& name, std::shared_ptr<TensorTransform> transform);
    PreprocessingPipeline& add_normalization(const std::vector<float>& mean, const std::vector<float>& std);
    PreprocessingPipeline& add_resize(size_t height, size_t width);
    

    std::shared_ptr<Tensor> process(const std::shared_ptr<Tensor>& input) const;
    

    bool has_transform(const std::string& name) const;
    void clear();
};

/**
 * @brief Factory functions for common transform combinations
 */
namespace factory {

    std::shared_ptr<PreprocessingPipeline> create_imagenet_pipeline(bool training = true);
    std::shared_ptr<PreprocessingPipeline> create_cifar10_pipeline(bool training = true);
    

    class SimpleBuilder {
    private:
        std::shared_ptr<PreprocessingPipeline> m_pipeline;
        
    public:
        SimpleBuilder();
        SimpleBuilder& resize(size_t height, size_t width);
        SimpleBuilder& random_horizontal_flip(float probability = 0.5f);
        SimpleBuilder& random_crop(size_t height, size_t width, size_t padding = 0);
        SimpleBuilder& normalize(const std::vector<float>& mean, const std::vector<float>& std);
        std::shared_ptr<PreprocessingPipeline> build();
    };
}

} // namespace transforms
} // namespace data
} // namespace dlvk
