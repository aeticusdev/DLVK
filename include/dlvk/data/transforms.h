#pragma once

#include "dlvk/tensor/tensor.h"
#include <random>
#include <functional>

namespace dlvk {
namespace data {
namespace transforms {

/**
 * @brief Base class for all data transformations
 */
class Transform {
public:
    virtual ~Transform() = default;
    virtual Tensor operator()(const Tensor& input) const = 0;
};

/**
 * @brief Normalize tensor values to specified mean and std
 */
class Normalize : public Transform {
private:
    std::vector<float> m_mean;
    std::vector<float> m_std;

public:
    Normalize(const std::vector<float>& mean, const std::vector<float>& std);
    Tensor operator()(const Tensor& input) const override;
};

/**
 * @brief Convert tensor to one-hot encoding
 */
class ToOneHot : public Transform {
private:
    size_t m_num_classes;

public:
    explicit ToOneHot(size_t num_classes);
    Tensor operator()(const Tensor& input) const override;
};

/**
 * @brief Random horizontal flip for images
 */
class RandomHorizontalFlip : public Transform {
private:
    float m_probability;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_dist;

public:
    explicit RandomHorizontalFlip(float probability = 0.5);
    Tensor operator()(const Tensor& input) const override;
};

/**
 * @brief Random vertical flip for images
 */
class RandomVerticalFlip : public Transform {
private:
    float m_probability;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_dist;

public:
    explicit RandomVerticalFlip(float probability = 0.5);
    Tensor operator()(const Tensor& input) const override;
};

/**
 * @brief Random rotation for images
 */
class RandomRotation : public Transform {
private:
    float m_max_angle; // in degrees
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_angle_dist;

public:
    explicit RandomRotation(float max_angle = 30.0f);
    Tensor operator()(const Tensor& input) const override;

private:
    Tensor rotate_image(const Tensor& input, float angle) const;
};

/**
 * @brief Add random noise to tensor
 */
class RandomNoise : public Transform {
private:
    float m_noise_factor;
    mutable std::mt19937 m_rng;
    mutable std::normal_distribution<float> m_noise_dist;

public:
    explicit RandomNoise(float noise_factor = 0.1f);
    Tensor operator()(const Tensor& input) const override;
};

/**
 * @brief Random scaling for images
 */
class RandomScale : public Transform {
private:
    float m_min_scale;
    float m_max_scale;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_scale_dist;

public:
    RandomScale(float min_scale = 0.8f, float max_scale = 1.2f);
    Tensor operator()(const Tensor& input) const override;

private:
    Tensor scale_image(const Tensor& input, float scale) const;
};

/**
 * @brief Compose multiple transformations
 */
class Compose : public Transform {
private:
    std::vector<std::shared_ptr<Transform>> m_transforms;

public:
    explicit Compose(std::vector<std::shared_ptr<Transform>> transforms);
    Tensor operator()(const Tensor& input) const override;
};

/**
 * @brief Factory functions for common transform combinations
 */
namespace factory {
    // MNIST transforms
    std::shared_ptr<Transform> mnist_train_transforms(float noise_factor = 0.05f);
    std::shared_ptr<Transform> mnist_test_transforms();
    
    // CIFAR-10 transforms
    std::shared_ptr<Transform> cifar10_train_transforms();
    std::shared_ptr<Transform> cifar10_test_transforms();
    
    // ImageNet transforms
    std::shared_ptr<Transform> imagenet_train_transforms();
    std::shared_ptr<Transform> imagenet_test_transforms();
}

} // namespace transforms
} // namespace data
} // namespace dlvk
