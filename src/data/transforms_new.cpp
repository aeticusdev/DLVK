#include "dlvk/data/transforms.h"
#include "dlvk/tensor/tensor_ops.h"
#include <cmath>
#include <algorithm>
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
    

    if (shape.size() >= 3) {
        size_t channels = shape[shape.size() - 3];
        size_t height = shape[shape.size() - 2];
        size_t width = shape[shape.size() - 1];
        
        for (size_t c = 0; c < channels && c < m_mean.size(); ++c) {
            float mean = m_mean[c];
            float std_val = m_std[c];
            
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t idx = c * height * width + h * width + w;
                    data[idx] = (data[idx] - mean) / std_val;
                }
            }
        }
    } else {

        if (!m_mean.empty() && !m_std.empty()) {
            float mean = m_mean[0];
            float std_val = m_std[0];
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = (data[i] - mean) / std_val;
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(shape, DataType::FLOAT32, input->device());
    result->upload_data(data.data());
    return result;
}

std::unique_ptr<Transform> Normalize::clone() const {
    return std::make_unique<Normalize>(m_mean, m_std);
}


std::shared_ptr<Tensor> Standardize::apply(const std::shared_ptr<Tensor>& input) const {
    std::vector<float> data(input->size());
    input->download_data(data.data());
    

    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    float mean = sum / data.size();
    

    float var_sum = 0.0f;
    for (float val : data) {
        var_sum += (val - mean) * (val - mean);
    }
    float std_val = std::sqrt(var_sum / data.size());
    

    if (std_val > 1e-8f) {  // Avoid division by zero
        for (float& val : data) {
            val = (val - mean) / std_val;
        }
    }
    
    auto result = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, input->device());
    result->upload_data(data.data());
    return result;
}

std::unique_ptr<Transform> Standardize::clone() const {
    return std::make_unique<Standardize>();
}


MinMaxScale::MinMaxScale(float min_val, float max_val) : m_min_val(min_val), m_max_val(max_val) {}

std::shared_ptr<Tensor> MinMaxScale::apply(const std::shared_ptr<Tensor>& input) const {
    std::vector<float> data(input->size());
    input->download_data(data.data());
    

    auto minmax = std::minmax_element(data.begin(), data.end());
    float data_min = *minmax.first;
    float data_max = *minmax.second;
    

    float range = data_max - data_min;
    if (range > 1e-8f) {  // Avoid division by zero
        float scale = (m_max_val - m_min_val) / range;
        for (float& val : data) {
            val = m_min_val + (val - data_min) * scale;
        }
    }
    
    auto result = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, input->device());
    result->upload_data(data.data());
    return result;
}

std::unique_ptr<Transform> MinMaxScale::clone() const {
    return std::make_unique<MinMaxScale>(m_min_val, m_max_val);
}


ToOneHot::ToOneHot(size_t num_classes) : m_num_classes(num_classes) {}

std::shared_ptr<Tensor> ToOneHot::apply(const std::shared_ptr<Tensor>& input) const {
    auto shape = input->shape();
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    

    std::vector<size_t> output_shape = shape;
    output_shape.push_back(m_num_classes);
    
    std::vector<float> output_data(input->size() * m_num_classes, 0.0f);
    
    for (size_t i = 0; i < input->size(); ++i) {
        int class_idx = static_cast<int>(input_data[i]);
        if (class_idx >= 0 && class_idx < static_cast<int>(m_num_classes)) {
            output_data[i * m_num_classes + class_idx] = 1.0f;
        }
    }
    
    auto result = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, input->device());
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> ToOneHot::clone() const {
    return std::make_unique<ToOneHot>(m_num_classes);
}


RandomHorizontalFlip::RandomHorizontalFlip(float probability) 
    : m_probability(probability), m_rng(std::random_device{}()), m_dist(0.0f, 1.0f) {}

std::shared_ptr<Tensor> RandomHorizontalFlip::apply(const std::shared_ptr<Tensor>& input) const {
    if (m_dist(m_rng) > m_probability) {
        return std::make_shared<Tensor>(*input);  // No flip
    }
    
    auto shape = input->shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Input must have at least 3 dimensions for image flipping");
    }
    
    size_t channels = shape[shape.size() - 3];
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<float> output_data(input->size());
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                size_t input_idx = c * height * width + h * width + w;
                size_t output_idx = c * height * width + h * width + (width - 1 - w);
                output_data[output_idx] = input_data[input_idx];
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(shape, DataType::FLOAT32, input->device());
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> RandomHorizontalFlip::clone() const {
    return std::make_unique<RandomHorizontalFlip>(m_probability);
}


RandomVerticalFlip::RandomVerticalFlip(float probability)
    : m_probability(probability), m_rng(std::random_device{}()), m_dist(0.0f, 1.0f) {}

std::shared_ptr<Tensor> RandomVerticalFlip::apply(const std::shared_ptr<Tensor>& input) const {
    if (m_dist(m_rng) > m_probability) {
        return std::make_shared<Tensor>(*input);  // No flip
    }
    
    auto shape = input->shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Input must have at least 3 dimensions for image flipping");
    }
    
    size_t channels = shape[shape.size() - 3];
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<float> output_data(input->size());
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                size_t input_idx = c * height * width + h * width + w;
                size_t output_idx = c * height * width + (height - 1 - h) * width + w;
                output_data[output_idx] = input_data[input_idx];
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(shape, DataType::FLOAT32, input->device());
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> RandomVerticalFlip::clone() const {
    return std::make_unique<RandomVerticalFlip>(m_probability);
}


RandomRotation::RandomRotation(float max_angle)
    : m_max_angle(max_angle), m_rng(std::random_device{}()), m_angle_dist(-max_angle, max_angle) {}

std::shared_ptr<Tensor> RandomRotation::apply(const std::shared_ptr<Tensor>& input) const {
    float angle = m_angle_dist(m_rng);
    return rotate_image(input, angle);
}

std::shared_ptr<Tensor> RandomRotation::rotate_image(const std::shared_ptr<Tensor>& input, float angle) const {

    auto shape = input->shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Input must have at least 3 dimensions for image rotation");
    }
    
    size_t channels = shape[shape.size() - 3];
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<float> output_data(input->size(), 0.0f);
    
    float angle_rad = angle * M_PI / 180.0f;
    float cos_angle = std::cos(angle_rad);
    float sin_angle = std::sin(angle_rad);
    
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {

                float x = w - center_x;
                float y = h - center_y;
                
                float rotated_x = x * cos_angle - y * sin_angle + center_x;
                float rotated_y = x * sin_angle + y * cos_angle + center_y;
                
                int src_x = static_cast<int>(std::round(rotated_x));
                int src_y = static_cast<int>(std::round(rotated_y));
                
                if (src_x >= 0 && src_x < static_cast<int>(width) && 
                    src_y >= 0 && src_y < static_cast<int>(height)) {
                    size_t src_idx = c * height * width + src_y * width + src_x;
                    size_t dst_idx = c * height * width + h * width + w;
                    output_data[dst_idx] = input_data[src_idx];
                }
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(shape, DataType::FLOAT32, input->device());
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> RandomRotation::clone() const {
    return std::make_unique<RandomRotation>(m_max_angle);
}


RandomCrop::RandomCrop(size_t crop_height, size_t crop_width, size_t padding)
    : m_crop_height(crop_height), m_crop_width(crop_width), m_padding(padding), m_rng(std::random_device{}()) {}

std::shared_ptr<Tensor> RandomCrop::apply(const std::shared_ptr<Tensor>& input) const {
    auto shape = input->shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Input must have at least 3 dimensions for cropping");
    }
    
    size_t channels = shape[shape.size() - 3];
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    

    size_t padded_height = height + 2 * m_padding;
    size_t padded_width = width + 2 * m_padding;
    
    if (m_crop_height > padded_height || m_crop_width > padded_width) {
        throw std::invalid_argument("Crop size is larger than padded input size");
    }
    

    std::uniform_int_distribution<size_t> h_dist(0, padded_height - m_crop_height);
    std::uniform_int_distribution<size_t> w_dist(0, padded_width - m_crop_width);
    
    size_t start_h = h_dist(m_rng);
    size_t start_w = w_dist(m_rng);
    
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<size_t> output_shape = shape;
    output_shape[output_shape.size() - 2] = m_crop_height;
    output_shape[output_shape.size() - 1] = m_crop_width;
    
    std::vector<float> output_data(channels * m_crop_height * m_crop_width);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < m_crop_height; ++h) {
            for (size_t w = 0; w < m_crop_width; ++w) {
                int src_h = static_cast<int>(start_h + h) - static_cast<int>(m_padding);
                int src_w = static_cast<int>(start_w + w) - static_cast<int>(m_padding);
                
                float value = 0.0f;
                if (src_h >= 0 && src_h < static_cast<int>(height) && 
                    src_w >= 0 && src_w < static_cast<int>(width)) {
                    size_t src_idx = c * height * width + src_h * width + src_w;
                    value = input_data[src_idx];
                }
                
                size_t dst_idx = c * m_crop_height * m_crop_width + h * m_crop_width + w;
                output_data[dst_idx] = value;
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, input->device());
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> RandomCrop::clone() const {
    return std::make_unique<RandomCrop>(m_crop_height, m_crop_width, m_padding);
}


CenterCrop::CenterCrop(size_t crop_height, size_t crop_width)
    : m_crop_height(crop_height), m_crop_width(crop_width) {}

std::shared_ptr<Tensor> CenterCrop::apply(const std::shared_ptr<Tensor>& input) const {
    auto shape = input->shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Input must have at least 3 dimensions for cropping");
    }
    
    size_t channels = shape[shape.size() - 3];
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    
    if (m_crop_height > height || m_crop_width > width) {
        throw std::invalid_argument("Crop size is larger than input size");
    }
    
    size_t start_h = (height - m_crop_height) / 2;
    size_t start_w = (width - m_crop_width) / 2;
    
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<size_t> output_shape = shape;
    output_shape[output_shape.size() - 2] = m_crop_height;
    output_shape[output_shape.size() - 1] = m_crop_width;
    
    std::vector<float> output_data(channels * m_crop_height * m_crop_width);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < m_crop_height; ++h) {
            for (size_t w = 0; w < m_crop_width; ++w) {
                size_t src_idx = c * height * width + (start_h + h) * width + (start_w + w);
                size_t dst_idx = c * m_crop_height * m_crop_width + h * m_crop_width + w;
                output_data[dst_idx] = input_data[src_idx];
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, input->device());
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> CenterCrop::clone() const {
    return std::make_unique<CenterCrop>(m_crop_height, m_crop_width);
}


RandomNoise::RandomNoise(float noise_factor)
    : m_noise_factor(noise_factor), m_rng(std::random_device{}()), m_noise_dist(0.0f, noise_factor) {}

std::shared_ptr<Tensor> RandomNoise::apply(const std::shared_ptr<Tensor>& input) const {
    std::vector<float> data(input->size());
    input->download_data(data.data());
    
    for (float& val : data) {
        val += m_noise_dist(m_rng);
    }
    
    auto result = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, input->device());
    result->upload_data(data.data());
    return result;
}

std::unique_ptr<Transform> RandomNoise::clone() const {
    return std::make_unique<RandomNoise>(m_noise_factor);
}


RandomBrightness::RandomBrightness(float brightness_factor)
    : m_brightness_factor(brightness_factor), m_rng(std::random_device{}()), 
      m_brightness_dist(-brightness_factor, brightness_factor) {}

std::shared_ptr<Tensor> RandomBrightness::apply(const std::shared_ptr<Tensor>& input) const {
    float brightness_change = m_brightness_dist(m_rng);
    
    std::vector<float> data(input->size());
    input->download_data(data.data());
    
    for (float& val : data) {
        val = std::clamp(val + brightness_change, 0.0f, 1.0f);
    }
    
    auto result = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, input->device());
    result->upload_data(data.data());
    return result;
}

std::unique_ptr<Transform> RandomBrightness::clone() const {
    return std::make_unique<RandomBrightness>(m_brightness_factor);
}


RandomContrast::RandomContrast(float contrast_range)
    : m_contrast_range(contrast_range), m_rng(std::random_device{}()),
      m_contrast_dist(1.0f - contrast_range, 1.0f + contrast_range) {}

std::shared_ptr<Tensor> RandomContrast::apply(const std::shared_ptr<Tensor>& input) const {
    float contrast_factor = m_contrast_dist(m_rng);
    
    std::vector<float> data(input->size());
    input->download_data(data.data());
    

    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    float mean = sum / data.size();
    
    for (float& val : data) {
        val = std::clamp(mean + contrast_factor * (val - mean), 0.0f, 1.0f);
    }
    
    auto result = std::make_shared<Tensor>(input->shape(), DataType::FLOAT32, input->device());
    result->upload_data(data.data());
    return result;
}

std::unique_ptr<Transform> RandomContrast::clone() const {
    return std::make_unique<RandomContrast>(m_contrast_range);
}


ColorJitter::ColorJitter(float brightness, float contrast, float saturation)
    : m_brightness(brightness), m_contrast(contrast), m_saturation(saturation), m_rng(std::random_device{}()) {}

std::shared_ptr<Tensor> ColorJitter::apply(const std::shared_ptr<Tensor>& input) const {
    auto result = input;
    

    if (m_brightness > 0) {
        auto brightness_transform = RandomBrightness(m_brightness);
        result = brightness_transform.apply(result);
    }
    

    if (m_contrast > 0) {
        auto contrast_transform = RandomContrast(m_contrast);
        result = contrast_transform.apply(result);
    }
    


    
    return result;
}

std::unique_ptr<Transform> ColorJitter::clone() const {
    return std::make_unique<ColorJitter>(m_brightness, m_contrast, m_saturation);
}


RandomScale::RandomScale(float min_scale, float max_scale)
    : m_min_scale(min_scale), m_max_scale(max_scale), m_rng(std::random_device{}()),
      m_scale_dist(min_scale, max_scale) {}

std::shared_ptr<Tensor> RandomScale::apply(const std::shared_ptr<Tensor>& input) const {
    float scale = m_scale_dist(m_rng);
    return scale_image(input, scale);
}

std::shared_ptr<Tensor> RandomScale::scale_image(const std::shared_ptr<Tensor>& input, float scale) const {
    auto shape = input->shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Input must have at least 3 dimensions for scaling");
    }
    
    size_t channels = shape[shape.size() - 3];
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    
    size_t new_height = static_cast<size_t>(height * scale);
    size_t new_width = static_cast<size_t>(width * scale);
    

    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<size_t> output_shape = shape;
    output_shape[output_shape.size() - 2] = new_height;
    output_shape[output_shape.size() - 1] = new_width;
    
    std::vector<float> output_data(channels * new_height * new_width);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < new_height; ++h) {
            for (size_t w = 0; w < new_width; ++w) {
                size_t src_h = static_cast<size_t>(h / scale);
                size_t src_w = static_cast<size_t>(w / scale);
                
                if (src_h < height && src_w < width) {
                    size_t src_idx = c * height * width + src_h * width + src_w;
                    size_t dst_idx = c * new_height * new_width + h * new_width + w;
                    output_data[dst_idx] = input_data[src_idx];
                }
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, input->device());
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> RandomScale::clone() const {
    return std::make_unique<RandomScale>(m_min_scale, m_max_scale);
}


Resize::Resize(size_t height, size_t width, InterpolationMode mode)
    : m_height(height), m_width(width), m_mode(mode) {}

std::shared_ptr<Tensor> Resize::apply(const std::shared_ptr<Tensor>& input) const {
    auto shape = input->shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Input must have at least 3 dimensions for resizing");
    }
    
    size_t channels = shape[shape.size() - 3];
    size_t input_height = shape[shape.size() - 2];
    size_t input_width = shape[shape.size() - 1];
    
    if (input_height == m_height && input_width == m_width) {
        return std::make_shared<Tensor>(*input);  // No resizing needed
    }
    
    std::vector<float> input_data(input->size());
    input->download_data(input_data.data());
    
    std::vector<size_t> output_shape = shape;
    output_shape[output_shape.size() - 2] = m_height;
    output_shape[output_shape.size() - 1] = m_width;
    
    std::vector<float> output_data(channels * m_height * m_width);
    
    float scale_h = static_cast<float>(input_height) / m_height;
    float scale_w = static_cast<float>(input_width) / m_width;
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < m_height; ++h) {
            for (size_t w = 0; w < m_width; ++w) {
                if (m_mode == InterpolationMode::NEAREST) {
                    size_t src_h = static_cast<size_t>(h * scale_h);
                    size_t src_w = static_cast<size_t>(w * scale_w);
                    
                    if (src_h < input_height && src_w < input_width) {
                        size_t src_idx = c * input_height * input_width + src_h * input_width + src_w;
                        size_t dst_idx = c * m_height * m_width + h * m_width + w;
                        output_data[dst_idx] = input_data[src_idx];
                    }
                } else {

                    float src_h_f = h * scale_h;
                    float src_w_f = w * scale_w;
                    
                    size_t src_h1 = static_cast<size_t>(src_h_f);
                    size_t src_w1 = static_cast<size_t>(src_w_f);
                    size_t src_h2 = std::min(src_h1 + 1, input_height - 1);
                    size_t src_w2 = std::min(src_w1 + 1, input_width - 1);
                    
                    float dh = src_h_f - src_h1;
                    float dw = src_w_f - src_w1;
                    
                    size_t idx11 = c * input_height * input_width + src_h1 * input_width + src_w1;
                    size_t idx12 = c * input_height * input_width + src_h1 * input_width + src_w2;
                    size_t idx21 = c * input_height * input_width + src_h2 * input_width + src_w1;
                    size_t idx22 = c * input_height * input_width + src_h2 * input_width + src_w2;
                    
                    float val11 = input_data[idx11];
                    float val12 = input_data[idx12];
                    float val21 = input_data[idx21];
                    float val22 = input_data[idx22];
                    
                    float interpolated = val11 * (1 - dh) * (1 - dw) +
                                       val12 * (1 - dh) * dw +
                                       val21 * dh * (1 - dw) +
                                       val22 * dh * dw;
                    
                    size_t dst_idx = c * m_height * m_width + h * m_width + w;
                    output_data[dst_idx] = interpolated;
                }
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(output_shape, DataType::FLOAT32, input->device());
    result->upload_data(output_data.data());
    return result;
}

std::unique_ptr<Transform> Resize::clone() const {
    return std::make_unique<Resize>(m_height, m_width, m_mode);
}


Compose::Compose(std::vector<std::shared_ptr<Transform>> transforms) : m_transforms(std::move(transforms)) {}

std::shared_ptr<Tensor> Compose::apply(const std::shared_ptr<Tensor>& input) const {
    auto result = input;
    for (const auto& transform : m_transforms) {
        if (auto tensor_transform = std::dynamic_pointer_cast<TensorTransform>(transform)) {
            result = tensor_transform->apply(result);
        } else {

            auto temp_tensor = (*transform)(*result);
            result = std::make_shared<Tensor>(temp_tensor);
        }
    }
    return result;
}

void Compose::add_transform(std::shared_ptr<Transform> transform) {
    m_transforms.push_back(transform);
}

std::unique_ptr<Transform> Compose::clone() const {
    std::vector<std::shared_ptr<Transform>> cloned_transforms;
    for (const auto& transform : m_transforms) {
        cloned_transforms.push_back(transform->clone());
    }
    return std::make_unique<Compose>(cloned_transforms);
}


RandomChoice::RandomChoice(std::vector<std::shared_ptr<Transform>> transforms)
    : m_transforms(std::move(transforms)), m_rng(std::random_device{}()),
      m_choice_dist(0, m_transforms.size() - 1) {}

std::shared_ptr<Tensor> RandomChoice::apply(const std::shared_ptr<Tensor>& input) const {
    if (m_transforms.empty()) {
        return std::make_shared<Tensor>(*input);
    }
    
    size_t choice = m_choice_dist(m_rng);
    auto& chosen_transform = m_transforms[choice];
    
    if (auto tensor_transform = std::dynamic_pointer_cast<TensorTransform>(chosen_transform)) {
        return tensor_transform->apply(input);
    } else {
        auto temp_tensor = (*chosen_transform)(*input);
        return std::make_shared<Tensor>(temp_tensor);
    }
}

std::unique_ptr<Transform> RandomChoice::clone() const {
    std::vector<std::shared_ptr<Transform>> cloned_transforms;
    for (const auto& transform : m_transforms) {
        cloned_transforms.push_back(transform->clone());
    }
    return std::make_unique<RandomChoice>(cloned_transforms);
}


RandomApply::RandomApply(std::shared_ptr<Transform> transform, float probability)
    : m_transform(transform), m_probability(probability), m_rng(std::random_device{}()),
      m_prob_dist(0.0f, 1.0f) {}

std::shared_ptr<Tensor> RandomApply::apply(const std::shared_ptr<Tensor>& input) const {
    if (m_prob_dist(m_rng) <= m_probability) {
        if (auto tensor_transform = std::dynamic_pointer_cast<TensorTransform>(m_transform)) {
            return tensor_transform->apply(input);
        } else {
            auto temp_tensor = (*m_transform)(*input);
            return std::make_shared<Tensor>(temp_tensor);
        }
    } else {
        return std::make_shared<Tensor>(*input);
    }
}

std::unique_ptr<Transform> RandomApply::clone() const {
    return std::make_unique<RandomApply>(m_transform->clone(), m_probability);
}


std::shared_ptr<Compose> LightAugmentation::create_transform() const {
    std::vector<std::shared_ptr<Transform>> transforms = {
        std::make_shared<RandomHorizontalFlip>(0.5f),
        std::make_shared<RandomBrightness>(0.1f)
    };
    return std::make_shared<Compose>(transforms);
}

std::shared_ptr<Compose> MediumAugmentation::create_transform() const {
    std::vector<std::shared_ptr<Transform>> transforms = {
        std::make_shared<RandomHorizontalFlip>(0.5f),
        std::make_shared<RandomVerticalFlip>(0.2f),
        std::make_shared<RandomRotation>(15.0f),
        std::make_shared<ColorJitter>(0.2f, 0.2f, 0.2f),
        std::make_shared<RandomNoise>(0.05f)
    };
    return std::make_shared<Compose>(transforms);
}

std::shared_ptr<Compose> HeavyAugmentation::create_transform() const {
    std::vector<std::shared_ptr<Transform>> transforms = {
        std::make_shared<RandomHorizontalFlip>(0.5f),
        std::make_shared<RandomVerticalFlip>(0.3f),
        std::make_shared<RandomRotation>(30.0f),
        std::make_shared<RandomScale>(0.7f, 1.3f),
        std::make_shared<ColorJitter>(0.4f, 0.4f, 0.4f),
        std::make_shared<RandomNoise>(0.1f),
        std::make_shared<RandomCrop>(224, 224, 4)
    };
    return std::make_shared<Compose>(transforms);
}


PreprocessingPipeline::PreprocessingPipeline() {
    m_transforms = std::make_shared<Compose>(std::vector<std::shared_ptr<Transform>>{});
}

PreprocessingPipeline& PreprocessingPipeline::add_transform(const std::string& name, std::shared_ptr<Transform> transform) {
    m_named_transforms[name] = transform;
    m_transforms->add_transform(transform);
    return *this;
}

PreprocessingPipeline& PreprocessingPipeline::add_normalization(const std::vector<float>& mean, const std::vector<float>& std) {
    auto normalize = std::make_shared<Normalize>(mean, std);
    return add_transform("normalize", normalize);
}

PreprocessingPipeline& PreprocessingPipeline::add_standardization() {
    auto standardize = std::make_shared<Standardize>();
    return add_transform("standardize", standardize);
}

PreprocessingPipeline& PreprocessingPipeline::add_min_max_scaling(float min_val, float max_val) {
    auto scale = std::make_shared<MinMaxScale>(min_val, max_val);
    return add_transform("min_max_scale", scale);
}

PreprocessingPipeline& PreprocessingPipeline::add_resize(size_t height, size_t width) {
    auto resize = std::make_shared<Resize>(height, width);
    return add_transform("resize", resize);
}

PreprocessingPipeline& PreprocessingPipeline::add_augmentation_strategy(std::shared_ptr<AugmentationStrategy> strategy) {
    auto transforms = strategy->create_transform();
    return add_transform("augmentation_" + strategy->name(), transforms);
}

std::shared_ptr<Tensor> PreprocessingPipeline::process(const std::shared_ptr<Tensor>& input) const {
    return m_transforms->apply(input);
}

bool PreprocessingPipeline::has_transform(const std::string& name) const {
    return m_named_transforms.find(name) != m_named_transforms.end();
}

void PreprocessingPipeline::remove_transform(const std::string& name) {
    auto it = m_named_transforms.find(name);
    if (it != m_named_transforms.end()) {
        m_named_transforms.erase(it);

    }
}

void PreprocessingPipeline::clear() {
    m_named_transforms.clear();
    m_transforms = std::make_shared<Compose>(std::vector<std::shared_ptr<Transform>>{});
}

} // namespace transforms
} // namespace data
} // namespace dlvk
