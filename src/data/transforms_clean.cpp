#include "dlvk/data/transforms_clean.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dlvk {
namespace data {
namespace transforms {


Tensor TensorTransform::operator()(const Tensor& input) const {
    auto input_ptr = std::make_shared<Tensor>(input);
    auto result = apply(input_ptr);
    return *result;
}


Normalize::Normalize(const std::vector<float>& mean, const std::vector<float>& std)
    : m_mean(mean), m_std(std) {
    if (mean.size() != std.size()) {
        throw std::invalid_argument("Mean and std vectors must have the same size");
    }
}

std::shared_ptr<Tensor> Normalize::apply(const std::shared_ptr<Tensor>& input) const {
    auto shape = input->shape();
    std::vector<float> data(input->size());
    input->download_data(data.data());
    

    for (size_t i = 0; i < data.size() && i < m_mean.size(); ++i) {
        data[i] = (data[i] - m_mean[i % m_mean.size()]) / m_std[i % m_std.size()];
    }
    
    auto result = std::make_shared<Tensor>(shape, DataType::FLOAT32, input->device());
    result->upload_data(data.data());
    return result;
}

std::unique_ptr<Transform> Normalize::clone() const {
    return std::make_unique<Normalize>(m_mean, m_std);
}


RandomHorizontalFlip::RandomHorizontalFlip(float probability) 
    : m_probability(probability), m_rng(std::random_device{}()), m_dist(0.0f, 1.0f) {}

std::shared_ptr<Tensor> RandomHorizontalFlip::apply(const std::shared_ptr<Tensor>& input) const {
    if (m_dist(m_rng) > m_probability) {
        return std::make_shared<Tensor>(*input);  // No flip
    }
    

    return std::make_shared<Tensor>(*input);
}

std::unique_ptr<Transform> RandomHorizontalFlip::clone() const {
    return std::make_unique<RandomHorizontalFlip>(m_probability);
}


RandomCrop::RandomCrop(size_t crop_height, size_t crop_width, size_t padding)
    : m_crop_height(crop_height), m_crop_width(crop_width), m_padding(padding), m_rng(std::random_device{}()) {}

std::shared_ptr<Tensor> RandomCrop::apply(const std::shared_ptr<Tensor>& input) const {
    auto shape = input->shape();
    if (shape.size() < 2) {
        return std::make_shared<Tensor>(*input);
    }
    

    std::vector<size_t> output_shape = shape;
    output_shape[output_shape.size() - 2] = m_crop_height;
    output_shape[output_shape.size() - 1] = m_crop_width;
    
    auto result = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, input->device());
    

    std::vector<float> dummy_data(result->size(), 0.5f);
    result->upload_data(dummy_data.data());
    
    return result;
}

std::unique_ptr<Transform> RandomCrop::clone() const {
    return std::make_unique<RandomCrop>(m_crop_height, m_crop_width, m_padding);
}


Resize::Resize(size_t height, size_t width, InterpolationMode mode)
    : m_height(height), m_width(width), m_mode(mode) {}

std::shared_ptr<Tensor> Resize::apply(const std::shared_ptr<Tensor>& input) const {
    auto shape = input->shape();
    if (shape.size() < 2) {
        return std::make_shared<Tensor>(*input);
    }
    

    std::vector<size_t> output_shape = shape;
    output_shape[output_shape.size() - 2] = m_height;
    output_shape[output_shape.size() - 1] = m_width;
    
    auto result = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, input->device());
    

    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<float> output_data(result->size());
    std::fill(output_data.begin(), output_data.end(), 0.5f);  // Simple demo implementation
    
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> Resize::clone() const {
    return std::make_unique<Resize>(m_height, m_width, m_mode);
}


Compose::Compose(std::vector<std::shared_ptr<TensorTransform>> transforms) : m_transforms(std::move(transforms)) {}

std::shared_ptr<Tensor> Compose::apply(const std::shared_ptr<Tensor>& input) const {
    auto result = input;
    for (const auto& transform : m_transforms) {
        result = transform->apply(result);
    }
    return result;
}

void Compose::add_transform(std::shared_ptr<TensorTransform> transform) {
    m_transforms.push_back(transform);
}

std::unique_ptr<Transform> Compose::clone() const {
    std::vector<std::shared_ptr<TensorTransform>> cloned_transforms;
    for (const auto& transform : m_transforms) {
        auto cloned = transform->clone();
        cloned_transforms.push_back(std::static_pointer_cast<TensorTransform>(std::shared_ptr<Transform>(cloned.release())));
    }
    return std::make_unique<Compose>(cloned_transforms);
}


std::shared_ptr<Compose> LightAugmentation::create_transform() const {
    std::vector<std::shared_ptr<TensorTransform>> transforms = {
        std::make_shared<RandomHorizontalFlip>(0.5f)
    };
    return std::make_shared<Compose>(transforms);
}


PreprocessingPipeline::PreprocessingPipeline() {
    m_transforms = std::make_shared<Compose>(std::vector<std::shared_ptr<TensorTransform>>{});
}

PreprocessingPipeline& PreprocessingPipeline::add_transform(const std::string& name, std::shared_ptr<TensorTransform> transform) {
    m_named_transforms[name] = transform;
    m_transforms->add_transform(transform);
    return *this;
}

PreprocessingPipeline& PreprocessingPipeline::add_normalization(const std::vector<float>& mean, const std::vector<float>& std) {
    auto normalize = std::make_shared<Normalize>(mean, std);
    return add_transform("normalize", normalize);
}

PreprocessingPipeline& PreprocessingPipeline::add_resize(size_t height, size_t width) {
    auto resize = std::make_shared<Resize>(height, width);
    return add_transform("resize", resize);
}

std::shared_ptr<Tensor> PreprocessingPipeline::process(const std::shared_ptr<Tensor>& input) const {
    return m_transforms->apply(input);
}

bool PreprocessingPipeline::has_transform(const std::string& name) const {
    return m_named_transforms.find(name) != m_named_transforms.end();
}

void PreprocessingPipeline::clear() {
    m_named_transforms.clear();
    m_transforms = std::make_shared<Compose>(std::vector<std::shared_ptr<TensorTransform>>{});
}


namespace factory {

std::shared_ptr<PreprocessingPipeline> create_imagenet_pipeline(bool training) {
    auto pipeline = std::make_shared<PreprocessingPipeline>();
    
    if (training) {
        pipeline->add_transform("random_crop", std::make_shared<RandomCrop>(224, 224, 4));
        pipeline->add_transform("random_flip", std::make_shared<RandomHorizontalFlip>(0.5f));
    } else {
        pipeline->add_resize(224, 224);
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


SimpleBuilder::SimpleBuilder() {
    m_pipeline = std::make_shared<PreprocessingPipeline>();
}

SimpleBuilder& SimpleBuilder::resize(size_t height, size_t width) {
    m_pipeline->add_resize(height, width);
    return *this;
}

SimpleBuilder& SimpleBuilder::random_horizontal_flip(float probability) {
    m_pipeline->add_transform("random_horizontal_flip", std::make_shared<RandomHorizontalFlip>(probability));
    return *this;
}

SimpleBuilder& SimpleBuilder::random_crop(size_t height, size_t width, size_t padding) {
    m_pipeline->add_transform("random_crop", std::make_shared<RandomCrop>(height, width, padding));
    return *this;
}

SimpleBuilder& SimpleBuilder::normalize(const std::vector<float>& mean, const std::vector<float>& std) {
    m_pipeline->add_normalization(mean, std);
    return *this;
}

std::shared_ptr<PreprocessingPipeline> SimpleBuilder::build() {
    return m_pipeline;
}

} // namespace factory
} // namespace transforms
} // namespace data
} // namespace dlvk
