#include "dlvk/data/transforms.h"
#include <cmath>
#include <algorithm>

namespace dlvk {
namespace data {
namespace transforms {

// Normalize implementation
Normalize::Normalize(const std::vector<float>& mean, const std::vector<float>& std)
    : m_mean(mean), m_std(std) {
    if (mean.size() != std.size()) {
        throw std::invalid_argument("Mean and std vectors must have the same size");
    }
}

Tensor Normalize::operator()(const Tensor& input) const {
    auto shape = input.shape();
    std::vector<float> data(input.data(), input.data() + input.size());
    
    // Assume input is in CHW format
    if (shape.size() >= 3) {
        size_t channels = shape[shape.size() - 3];
        size_t height = shape[shape.size() - 2];
        size_t width = shape[shape.size() - 1];
        
        for (size_t c = 0; c < channels && c < m_mean.size(); ++c) {
            float mean = m_mean[c];
            float std = m_std[c];
            
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t idx = c * height * width + h * width + w;
                    data[idx] = (data[idx] - mean) / std;
                }
            }
        }
    } else {
        // For 1D or 2D tensors, normalize with first mean/std
        if (!m_mean.empty() && !m_std.empty()) {
            float mean = m_mean[0];
            float std = m_std[0];
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = (data[i] - mean) / std;
            }
        }
    }
    
    return Tensor(shape, data.data());
}

// ToOneHot implementation
ToOneHot::ToOneHot(size_t num_classes) : m_num_classes(num_classes) {}

Tensor ToOneHot::operator()(const Tensor& input) const {
    if (input.size() != 1) {
        throw std::invalid_argument("ToOneHot expects a single scalar value");
    }
    
    int class_idx = static_cast<int>(input.data()[0]);
    if (class_idx < 0 || class_idx >= static_cast<int>(m_num_classes)) {
        throw std::out_of_range("Class index out of range");
    }
    
    std::vector<float> one_hot(m_num_classes, 0.0f);
    one_hot[class_idx] = 1.0f;
    
    return Tensor({m_num_classes}, one_hot.data());
}

// RandomHorizontalFlip implementation
RandomHorizontalFlip::RandomHorizontalFlip(float probability)
    : m_probability(probability)
    , m_rng(std::random_device{}())
    , m_dist(0.0f, 1.0f) {}

Tensor RandomHorizontalFlip::operator()(const Tensor& input) const {
    if (m_dist(m_rng) > m_probability) {
        return input; // No flip
    }
    
    auto shape = input.shape();
    if (shape.size() < 2) {
        return input; // Can't flip 1D tensor
    }
    
    std::vector<float> data(input.data(), input.data() + input.size());
    
    // Assume last two dimensions are height and width
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    size_t channel_size = height * width;
    size_t num_channels = input.size() / channel_size;
    
    for (size_t c = 0; c < num_channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width / 2; ++w) {
                size_t left_idx = c * channel_size + h * width + w;
                size_t right_idx = c * channel_size + h * width + (width - 1 - w);
                std::swap(data[left_idx], data[right_idx]);
            }
        }
    }
    
    return Tensor(shape, data.data());
}

// RandomVerticalFlip implementation
RandomVerticalFlip::RandomVerticalFlip(float probability)
    : m_probability(probability)
    , m_rng(std::random_device{}())
    , m_dist(0.0f, 1.0f) {}

Tensor RandomVerticalFlip::operator()(const Tensor& input) const {
    if (m_dist(m_rng) > m_probability) {
        return input; // No flip
    }
    
    auto shape = input.shape();
    if (shape.size() < 2) {
        return input; // Can't flip 1D tensor
    }
    
    std::vector<float> data(input.data(), input.data() + input.size());
    
    // Assume last two dimensions are height and width
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    size_t channel_size = height * width;
    size_t num_channels = input.size() / channel_size;
    
    for (size_t c = 0; c < num_channels; ++c) {
        for (size_t h = 0; h < height / 2; ++h) {
            for (size_t w = 0; w < width; ++w) {
                size_t top_idx = c * channel_size + h * width + w;
                size_t bottom_idx = c * channel_size + (height - 1 - h) * width + w;
                std::swap(data[top_idx], data[bottom_idx]);
            }
        }
    }
    
    return Tensor(shape, data.data());
}

// RandomRotation implementation
RandomRotation::RandomRotation(float max_angle)
    : m_max_angle(max_angle)
    , m_rng(std::random_device{}())
    , m_angle_dist(-max_angle, max_angle) {}

Tensor RandomRotation::operator()(const Tensor& input) const {
    float angle = m_angle_dist(m_rng);
    return rotate_image(input, angle);
}

Tensor RandomRotation::rotate_image(const Tensor& input, float angle) const {
    // Simple rotation implementation using nearest neighbor interpolation
    auto shape = input.shape();
    if (shape.size() < 2) {
        return input;
    }
    
    size_t height = shape[shape.size() - 2];
    size_t width = shape[shape.size() - 1];
    size_t channel_size = height * width;
    size_t num_channels = input.size() / channel_size;
    
    std::vector<float> output_data(input.size(), 0.0f);
    
    float angle_rad = angle * M_PI / 180.0f;
    float cos_a = std::cos(angle_rad);
    float sin_a = std::sin(angle_rad);
    
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;
    
    for (size_t c = 0; c < num_channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                // Rotate coordinates
                float dx = x - center_x;
                float dy = y - center_y;
                int src_x = static_cast<int>(cos_a * dx + sin_a * dy + center_x);
                int src_y = static_cast<int>(-sin_a * dx + cos_a * dy + center_y);
                
                // Check bounds and copy pixel
                if (src_x >= 0 && src_x < static_cast<int>(width) && 
                    src_y >= 0 && src_y < static_cast<int>(height)) {
                    size_t src_idx = c * channel_size + src_y * width + src_x;
                    size_t dst_idx = c * channel_size + y * width + x;
                    output_data[dst_idx] = input.data()[src_idx];
                }
            }
        }
    }
    
    return Tensor(shape, output_data.data());
}

// RandomNoise implementation
RandomNoise::RandomNoise(float noise_factor)
    : m_noise_factor(noise_factor)
    , m_rng(std::random_device{}())
    , m_noise_dist(0.0f, noise_factor) {}

Tensor RandomNoise::operator()(const Tensor& input) const {
    std::vector<float> data(input.data(), input.data() + input.size());
    
    for (float& value : data) {
        value += m_noise_dist(m_rng);
        value = std::clamp(value, 0.0f, 1.0f); // Keep in valid range
    }
    
    return Tensor(input.shape(), data.data());
}

// RandomScale implementation
RandomScale::RandomScale(float min_scale, float max_scale)
    : m_min_scale(min_scale)
    , m_max_scale(max_scale)
    , m_rng(std::random_device{}())
    , m_scale_dist(min_scale, max_scale) {}

Tensor RandomScale::operator()(const Tensor& input) const {
    float scale = m_scale_dist(m_rng);
    return scale_image(input, scale);
}

Tensor RandomScale::scale_image(const Tensor& input, float scale) const {
    // Simple scaling implementation
    auto shape = input.shape();
    if (shape.size() < 2) {
        return input;
    }
    
    std::vector<float> data(input.data(), input.data() + input.size());
    
    // Simple scaling by multiplying values
    for (float& value : data) {
        value *= scale;
        value = std::clamp(value, 0.0f, 1.0f);
    }
    
    return Tensor(shape, data.data());
}

// Compose implementation
Compose::Compose(std::vector<std::shared_ptr<Transform>> transforms)
    : m_transforms(std::move(transforms)) {}

Tensor Compose::operator()(const Tensor& input) const {
    Tensor result = input;
    for (const auto& transform : m_transforms) {
        result = (*transform)(result);
    }
    return result;
}

// Factory functions
namespace factory {

std::shared_ptr<Transform> mnist_train_transforms(float noise_factor) {
    return std::make_shared<Compose>(std::vector<std::shared_ptr<Transform>>{
        std::make_shared<RandomHorizontalFlip>(0.5f),
        std::make_shared<RandomRotation>(15.0f),
        std::make_shared<RandomNoise>(noise_factor),
        std::make_shared<Normalize>(std::vector<float>{0.1307f}, std::vector<float>{0.3081f})
    });
}

std::shared_ptr<Transform> mnist_test_transforms() {
    return std::make_shared<Normalize>(std::vector<float>{0.1307f}, std::vector<float>{0.3081f});
}

std::shared_ptr<Transform> cifar10_train_transforms() {
    return std::make_shared<Compose>(std::vector<std::shared_ptr<Transform>>{
        std::make_shared<RandomHorizontalFlip>(0.5f),
        std::make_shared<RandomRotation>(10.0f),
        std::make_shared<Normalize>(
            std::vector<float>{0.4914f, 0.4822f, 0.4465f},
            std::vector<float>{0.2023f, 0.1994f, 0.2010f}
        )
    });
}

std::shared_ptr<Transform> cifar10_test_transforms() {
    return std::make_shared<Normalize>(
        std::vector<float>{0.4914f, 0.4822f, 0.4465f},
        std::vector<float>{0.2023f, 0.1994f, 0.2010f}
    );
}

std::shared_ptr<Transform> imagenet_train_transforms() {
    return std::make_shared<Compose>(std::vector<std::shared_ptr<Transform>>{
        std::make_shared<RandomHorizontalFlip>(0.5f),
        std::make_shared<RandomRotation>(10.0f),
        std::make_shared<RandomScale>(0.8f, 1.2f),
        std::make_shared<Normalize>(
            std::vector<float>{0.485f, 0.456f, 0.406f},
            std::vector<float>{0.229f, 0.224f, 0.225f}
        )
    });
}

std::shared_ptr<Transform> imagenet_test_transforms() {
    return std::make_shared<Normalize>(
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );
}

} // namespace factory

} // namespace transforms
} // namespace data
} // namespace dlvk
