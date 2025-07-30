#pragma once

#include "dlvk/tensor/tensor.h"
#include <random>
#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>

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
    virtual std::unique_ptr<Transform> clone() const = 0;
};

/**
 * @brief Tensor-integrated transformation interface
 */
class TensorTransform : public Transform {
public:
    virtual ~TensorTransform() = default;
    virtual std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const = 0;
    virtual Tensor operator()(const Tensor& input) const override final;
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
 * @brief Standardize tensor values (zero mean, unit variance)
 */
class Standardize : public TensorTransform {
public:
    Standardize() = default;
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Feature scaling (min-max normalization)
 */
class MinMaxScale : public TensorTransform {
private:
    float m_min_val;
    float m_max_val;

public:
    MinMaxScale(float min_val = 0.0f, float max_val = 1.0f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Convert tensor to one-hot encoding
 */
class ToOneHot : public TensorTransform {
private:
    size_t m_num_classes;

public:
    explicit ToOneHot(size_t num_classes);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random horizontal flip for images
 */
class RandomHorizontalFlip : public TensorTransform {
private:
    float m_probability;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_dist;

public:
    explicit RandomHorizontalFlip(float probability = 0.5);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random vertical flip for images
 */
class RandomVerticalFlip : public TensorTransform {
private:
    float m_probability;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_dist;

public:
    explicit RandomVerticalFlip(float probability = 0.5);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random rotation for images
 */
class RandomRotation : public TensorTransform {
private:
    float m_max_angle; // in degrees
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_angle_dist;

public:
    explicit RandomRotation(float max_angle = 30.0f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;

private:
    std::shared_ptr<Tensor> rotate_image(const std::shared_ptr<Tensor>& input, float angle) const;
};

/**
 * @brief Random cropping for images
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
 * @brief Center cropping for images
 */
class CenterCrop : public TensorTransform {
private:
    size_t m_crop_height;
    size_t m_crop_width;

public:
    CenterCrop(size_t crop_height, size_t crop_width);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};
/**
 * @brief Add random noise to tensor
 */
class RandomNoise : public TensorTransform {
private:
    float m_noise_factor;
    mutable std::mt19937 m_rng;
    mutable std::normal_distribution<float> m_noise_dist;

public:
    explicit RandomNoise(float noise_factor = 0.1f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random brightness adjustment
 */
class RandomBrightness : public TensorTransform {
private:
    float m_brightness_factor;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_brightness_dist;

public:
    explicit RandomBrightness(float brightness_factor = 0.2f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random contrast adjustment
 */
class RandomContrast : public TensorTransform {
private:
    float m_contrast_range;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_contrast_dist;

public:
    explicit RandomContrast(float contrast_range = 0.3f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random color jitter (brightness, contrast, saturation)
 */
class ColorJitter : public TensorTransform {
private:
    float m_brightness;
    float m_contrast;
    float m_saturation;
    mutable std::mt19937 m_rng;

public:
    ColorJitter(float brightness = 0.2f, float contrast = 0.2f, float saturation = 0.2f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Random scaling for images
 */
class RandomScale : public TensorTransform {
private:
    float m_min_scale;
    float m_max_scale;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_scale_dist;

public:
    RandomScale(float min_scale = 0.8f, float max_scale = 1.2f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;

private:
    std::shared_ptr<Tensor> scale_image(const std::shared_ptr<Tensor>& input, float scale) const;
};

/**
 * @brief Resize images to specified dimensions
 */
class Resize : public TensorTransform {
private:
    size_t m_height;
    size_t m_width;
    enum class InterpolationMode { NEAREST, BILINEAR } m_mode;

public:
    Resize(size_t height, size_t width, InterpolationMode mode = InterpolationMode::BILINEAR);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};
};

/**
 * @brief Compose multiple transformations
 */
class Compose : public TensorTransform {
private:
    std::vector<std::shared_ptr<Transform>> m_transforms;

public:
    explicit Compose(std::vector<std::shared_ptr<Transform>> transforms);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
    

    void add_transform(std::shared_ptr<Transform> transform);
    size_t size() const { return m_transforms.size(); }
};

/**
 * @brief Random choice between multiple transforms
 */
class RandomChoice : public TensorTransform {
private:
    std::vector<std::shared_ptr<Transform>> m_transforms;
    mutable std::mt19937 m_rng;
    mutable std::uniform_int_distribution<size_t> m_choice_dist;

public:
    explicit RandomChoice(std::vector<std::shared_ptr<Transform>> transforms);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Apply transforms with specified probabilities
 */
class RandomApply : public TensorTransform {
private:
    std::shared_ptr<Transform> m_transform;
    float m_probability;
    mutable std::mt19937 m_rng;
    mutable std::uniform_real_distribution<float> m_prob_dist;

public:
    RandomApply(std::shared_ptr<Transform> transform, float probability = 0.5f);
    std::shared_ptr<Tensor> apply(const std::shared_ptr<Tensor>& input) const override;
    std::unique_ptr<Transform> clone() const override;
};

/**
 * @brief Augmentation strategy interface
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
    std::string name() const override { return "light"; }
};

/**
 * @brief Medium augmentation strategy
 */
class MediumAugmentation : public AugmentationStrategy {
public:
    std::shared_ptr<Compose> create_transform() const override;
    std::string name() const override { return "medium"; }
};

/**
 * @brief Heavy augmentation strategy
 */
class HeavyAugmentation : public AugmentationStrategy {
public:
    std::shared_ptr<Compose> create_transform() const override;
    std::string name() const override { return "heavy"; }
};

/**
 * @brief Custom preprocessing pipeline
 */
class PreprocessingPipeline {
private:
    std::shared_ptr<Compose> m_transforms;
    std::unordered_map<std::string, std::shared_ptr<Transform>> m_named_transforms;

public:
    PreprocessingPipeline();
    

    PreprocessingPipeline& add_transform(const std::string& name, std::shared_ptr<Transform> transform);
    PreprocessingPipeline& add_normalization(const std::vector<float>& mean, const std::vector<float>& std);
    PreprocessingPipeline& add_standardization();
    PreprocessingPipeline& add_min_max_scaling(float min_val = 0.0f, float max_val = 1.0f);
    PreprocessingPipeline& add_resize(size_t height, size_t width);
    PreprocessingPipeline& add_augmentation_strategy(std::shared_ptr<AugmentationStrategy> strategy);
    

    std::shared_ptr<Tensor> process(const std::shared_ptr<Tensor>& input) const;
    

    bool has_transform(const std::string& name) const;
    void remove_transform(const std::string& name);
    void clear();
};

/**
 * @brief Factory functions for common transform combinations
 */
namespace factory {

    std::shared_ptr<PreprocessingPipeline> create_imagenet_pipeline(bool training = true);
    std::shared_ptr<PreprocessingPipeline> create_cifar10_pipeline(bool training = true);
    std::shared_ptr<PreprocessingPipeline> create_mnist_pipeline(bool training = true);
    std::shared_ptr<PreprocessingPipeline> create_coco_pipeline(bool training = true);
    

    std::shared_ptr<PreprocessingPipeline> create_classification_pipeline(size_t input_size, bool training = true);
    std::shared_ptr<PreprocessingPipeline> create_segmentation_pipeline(size_t input_size, bool training = true);
    std::shared_ptr<PreprocessingPipeline> create_detection_pipeline(size_t input_size, bool training = true);
    

    class PreprocessingPipelineBuilder {
    private:
        std::shared_ptr<PreprocessingPipeline> m_pipeline;
        
    public:
        PreprocessingPipelineBuilder();
        

        PreprocessingPipelineBuilder& resize(size_t height, size_t width);
        PreprocessingPipelineBuilder& random_crop(size_t height, size_t width, size_t padding = 0);
        PreprocessingPipelineBuilder& center_crop(size_t height, size_t width);
        

        PreprocessingPipelineBuilder& random_horizontal_flip(float probability = 0.5f);
        PreprocessingPipelineBuilder& random_vertical_flip(float probability = 0.5f);
        PreprocessingPipelineBuilder& random_rotation(float max_angle = 15.0f);
        PreprocessingPipelineBuilder& random_scale(float min_scale = 0.8f, float max_scale = 1.2f);
        PreprocessingPipelineBuilder& color_jitter(float brightness = 0.2f, float contrast = 0.2f, float saturation = 0.2f);
        PreprocessingPipelineBuilder& random_noise(float noise_factor = 0.05f);
        

        PreprocessingPipelineBuilder& normalize(const std::vector<float>& mean, const std::vector<float>& std);
        PreprocessingPipelineBuilder& standardize();
        PreprocessingPipelineBuilder& min_max_scale(float min_val = 0.0f, float max_val = 1.0f);
        PreprocessingPipelineBuilder& to_one_hot(size_t num_classes);
        

        PreprocessingPipelineBuilder& augmentation_strategy(std::shared_ptr<AugmentationStrategy> strategy);
        PreprocessingPipelineBuilder& light_augmentation();
        PreprocessingPipelineBuilder& medium_augmentation();
        PreprocessingPipelineBuilder& heavy_augmentation();
        

        PreprocessingPipelineBuilder& custom_transform(const std::string& name, std::shared_ptr<Transform> transform);
        

        std::shared_ptr<PreprocessingPipeline> build();
    };
}

} // namespace transforms
} // namespace data
} // namespace dlvk
