#include "dlvk/optimizers/optimizers.h"
#include "dlvk/tensor/tensor.h"
#include <iostream>
#include <cmath>

namespace dlvk {
namespace GradientClipping {

float compute_grad_norm(const std::vector<std::shared_ptr<Tensor>>& gradients) {
    float total_norm = 0.0f;
    
    for (const auto& grad : gradients) {
        if (!grad) {
            continue;
        }
        
        // Download gradient data
        std::vector<float> grad_data(grad->size());
        grad->download_data(grad_data.data());
        
        // Compute squared norm
        for (float val : grad_data) {
            total_norm += val * val;
        }
    }
    
    return std::sqrt(total_norm);
}

void clip_grad_norm(std::vector<std::shared_ptr<Tensor>>& gradients, float max_norm) {
    float total_norm = compute_grad_norm(gradients);
    
    if (total_norm <= max_norm) {
        return; // No clipping needed
    }
    
    float clip_coef = max_norm / total_norm;
    
    // Scale all gradients
    for (auto& grad : gradients) {
        if (!grad) {
            continue;
        }
        
        // Download, scale, and upload gradient data
        std::vector<float> grad_data(grad->size());
        grad->download_data(grad_data.data());
        
        for (float& val : grad_data) {
            val *= clip_coef;
        }
        
        grad->upload_data(grad_data.data());
    }
}

void clip_grad_value(std::vector<std::shared_ptr<Tensor>>& gradients, float min_value, float max_value) {
    for (auto& grad : gradients) {
        if (!grad) {
            continue;
        }
        
        // Download gradient data
        std::vector<float> grad_data(grad->size());
        grad->download_data(grad_data.data());
        
        // Clip values
        for (float& val : grad_data) {
            val = std::max(min_value, std::min(max_value, val));
        }
        
        // Upload clipped data
        grad->upload_data(grad_data.data());
    }
}

} // namespace GradientClipping
} // namespace dlvk
