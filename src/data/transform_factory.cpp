#include "dlvk/data/transforms.h"

namespace dlvk {
namespace data {
namespace transforms {
namespace factory {


std::shared_ptr<PreprocessingPipeline> create_imagenet_pipeline(bool training) {
    auto pipeline = std::make_shared<PreprocessingPipeline>();
    
    if (training) {

        pipeline->add_resize(256, 256);
        pipeline->add_transform("random_crop", std::make_shared<RandomCrop>(224, 224, 4));
        pipeline->add_transform("random_flip", std::make_shared<RandomHorizontalFlip>(0.5f));
        pipeline->add_transform("color_jitter", std::make_shared<ColorJitter>(0.4f, 0.4f, 0.4f));
    } else {

        pipeline->add_resize(224, 224);
        pipeline->add_transform("center_crop", std::make_shared<CenterCrop>(224, 224));
    }
    

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    pipeline->add_normalization(mean, std);
    
    return pipeline;
}

std::shared_ptr<PreprocessingPipeline> create_cifar10_pipeline(bool training) {
    auto pipeline = std::make_shared<PreprocessingPipeline>();
    
    if (training) {

        pipeline->add_transform("random_crop", std::make_shared<RandomCrop>(32, 32, 4));
        pipeline->add_transform("random_flip", std::make_shared<RandomHorizontalFlip>(0.5f));
    }
    

    std::vector<float> mean = {0.4914f, 0.4822f, 0.4465f};
    std::vector<float> std = {0.2023f, 0.1994f, 0.2010f};
    pipeline->add_normalization(mean, std);
    
    return pipeline;
}

std::shared_ptr<PreprocessingPipeline> create_mnist_pipeline(bool training) {
    auto pipeline = std::make_shared<PreprocessingPipeline>();
    
    if (training) {

        pipeline->add_transform("random_rotation", std::make_shared<RandomRotation>(10.0f));
        pipeline->add_transform("random_crop", std::make_shared<RandomCrop>(28, 28, 2));
    }
    

    std::vector<float> mean = {0.1307f};
    std::vector<float> std = {0.3081f};
    pipeline->add_normalization(mean, std);
    
    return pipeline;
}

std::shared_ptr<PreprocessingPipeline> create_coco_pipeline(bool training) {
    auto pipeline = std::make_shared<PreprocessingPipeline>();
    
    if (training) {

        pipeline->add_transform("random_scale", std::make_shared<RandomScale>(0.8f, 1.2f));
        pipeline->add_transform("random_flip", std::make_shared<RandomHorizontalFlip>(0.5f));
        pipeline->add_transform("color_jitter", std::make_shared<ColorJitter>(0.1f, 0.1f, 0.1f));
    }
    
    pipeline->add_resize(416, 416);  // Common YOLO input size
    

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    pipeline->add_normalization(mean, std);
    
    return pipeline;
}


std::shared_ptr<PreprocessingPipeline> create_classification_pipeline(size_t input_size, bool training) {
    auto pipeline = std::make_shared<PreprocessingPipeline>();
    
    if (training) {

        pipeline->add_resize(static_cast<size_t>(input_size * 1.15), static_cast<size_t>(input_size * 1.15));
        pipeline->add_transform("random_crop", std::make_shared<RandomCrop>(input_size, input_size));
        pipeline->add_transform("random_flip", std::make_shared<RandomHorizontalFlip>(0.5f));
        pipeline->add_transform("color_jitter", std::make_shared<ColorJitter>(0.3f, 0.3f, 0.3f));
        pipeline->add_transform("random_rotation", std::make_shared<RandomRotation>(15.0f));
    } else {

        pipeline->add_resize(input_size, input_size);
    }
    
    pipeline->add_standardization();
    return pipeline;
}

std::shared_ptr<PreprocessingPipeline> create_segmentation_pipeline(size_t input_size, bool training) {
    auto pipeline = std::make_shared<PreprocessingPipeline>();
    
    if (training) {

        pipeline->add_transform("random_scale", std::make_shared<RandomScale>(0.5f, 2.0f));
        pipeline->add_transform("random_crop", std::make_shared<RandomCrop>(input_size, input_size));
        pipeline->add_transform("random_flip", std::make_shared<RandomHorizontalFlip>(0.5f));
        pipeline->add_transform("color_jitter", std::make_shared<ColorJitter>(0.2f, 0.2f, 0.2f));
    } else {
        pipeline->add_resize(input_size, input_size);
    }
    

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    pipeline->add_normalization(mean, std);
    
    return pipeline;
}

std::shared_ptr<PreprocessingPipeline> create_detection_pipeline(size_t input_size, bool training) {
    auto pipeline = std::make_shared<PreprocessingPipeline>();
    
    if (training) {

        pipeline->add_transform("random_scale", std::make_shared<RandomScale>(0.8f, 1.2f));
        pipeline->add_transform("random_flip", std::make_shared<RandomHorizontalFlip>(0.5f));
        pipeline->add_transform("color_jitter", std::make_shared<ColorJitter>(0.1f, 0.1f, 0.1f));
        pipeline->add_transform("random_brightness", std::make_shared<RandomBrightness>(0.1f));
    }
    
    pipeline->add_resize(input_size, input_size);
    

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    pipeline->add_normalization(mean, std);
    
    return pipeline;
}


factory::PreprocessingPipelineBuilder::PreprocessingPipelineBuilder() {
    m_pipeline = std::make_shared<PreprocessingPipeline>();
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::resize(size_t height, size_t width) {
    m_pipeline->add_resize(height, width);
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::random_crop(size_t height, size_t width, size_t padding) {
    m_pipeline->add_transform("random_crop", std::make_shared<RandomCrop>(height, width, padding));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::center_crop(size_t height, size_t width) {
    m_pipeline->add_transform("center_crop", std::make_shared<CenterCrop>(height, width));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::random_horizontal_flip(float probability) {
    m_pipeline->add_transform("random_horizontal_flip", std::make_shared<RandomHorizontalFlip>(probability));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::random_vertical_flip(float probability) {
    m_pipeline->add_transform("random_vertical_flip", std::make_shared<RandomVerticalFlip>(probability));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::random_rotation(float max_angle) {
    m_pipeline->add_transform("random_rotation", std::make_shared<RandomRotation>(max_angle));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::random_scale(float min_scale, float max_scale) {
    m_pipeline->add_transform("random_scale", std::make_shared<RandomScale>(min_scale, max_scale));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::color_jitter(float brightness, float contrast, float saturation) {
    m_pipeline->add_transform("color_jitter", std::make_shared<ColorJitter>(brightness, contrast, saturation));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::random_noise(float noise_factor) {
    m_pipeline->add_transform("random_noise", std::make_shared<RandomNoise>(noise_factor));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::normalize(const std::vector<float>& mean, const std::vector<float>& std) {
    m_pipeline->add_normalization(mean, std);
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::standardize() {
    m_pipeline->add_standardization();
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::min_max_scale(float min_val, float max_val) {
    m_pipeline->add_min_max_scaling(min_val, max_val);
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::to_one_hot(size_t num_classes) {
    m_pipeline->add_transform("to_one_hot", std::make_shared<ToOneHot>(num_classes));
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::augmentation_strategy(std::shared_ptr<AugmentationStrategy> strategy) {
    m_pipeline->add_augmentation_strategy(strategy);
    return *this;
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::light_augmentation() {
    return augmentation_strategy(std::make_shared<LightAugmentation>());
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::medium_augmentation() {
    return augmentation_strategy(std::make_shared<MediumAugmentation>());
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::heavy_augmentation() {
    return augmentation_strategy(std::make_shared<HeavyAugmentation>());
}

factory::PreprocessingPipelineBuilder& factory::PreprocessingPipelineBuilder::custom_transform(const std::string& name, std::shared_ptr<Transform> transform) {
    m_pipeline->add_transform(name, transform);
    return *this;
}

std::shared_ptr<PreprocessingPipeline> factory::PreprocessingPipelineBuilder::build() {
    return m_pipeline;
}

} // namespace factory
} // namespace transforms
} // namespace data
} // namespace dlvk
