#pragma once

// DLVK - Vulkan Machine Learning Framework
// Main header file for easy inclusion

// Core components
#include "dlvk/core/vulkan_device.h"

// Tensor operations
#include "dlvk/tensor/tensor.h"

// Neural network layers
#include "dlvk/layers/layer.h"

// Common aliases and utilities
namespace dlvk {
    // Version information
    constexpr int VERSION_MAJOR = 0;
    constexpr int VERSION_MINOR = 1;
    constexpr int VERSION_PATCH = 0;
    
    // Common tensor shapes for convenience
    using Shape = std::vector<size_t>;
    
    // Create common tensor shapes
    inline Shape make_shape(std::initializer_list<size_t> dims) {
        return Shape(dims);
    }
    
    // Utility functions
    inline std::string version_string() {
        return std::to_string(VERSION_MAJOR) + "." + 
               std::to_string(VERSION_MINOR) + "." + 
               std::to_string(VERSION_PATCH);
    }
}

// Convenience macros
#define DLVK_VERSION_STRING dlvk::version_string()
