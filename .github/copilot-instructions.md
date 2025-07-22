<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# DLVK - Vulkan Machine Learning Framework

This is a high-performance machine learning framework built on top of Vulkan compute shaders for cross-platform GPU acceleration.

## Project Structure
- `src/core/` - Core framework components (Device, Memory, Pipeline management)
- `src/compute/` - Vulkan compute shader implementations
- `src/tensor/` - Tensor operations and data structures
- `src/layers/` - Neural network layer implementations
- `src/optimizers/` - Optimization algorithms
- `examples/` - Example usage and demos
- `shaders/` - GLSL compute shaders
- `tests/` - Unit and integration tests

## Development Guidelines
- Use modern C++17/20 features
- Follow RAII patterns for Vulkan resource management
- Implement proper error handling with VkResult checking
- Use compute shaders for parallel operations
- Optimize memory usage with buffer pools
- Ensure cross-platform compatibility (Windows, Linux, macOS)

## Key Components
- VulkanDevice: Manages Vulkan device and queue families
- Tensor: Multi-dimensional array with GPU memory backing
- ComputePipeline: Wrapper for Vulkan compute pipelines
- Layer: Base class for neural network layers
- Optimizer: Base class for optimization algorithms

## Coding Standards
- Use snake_case for variables and functions
- Use PascalCase for classes and structs
- Prefix member variables with m_
- Use descriptive names for shaders and pipeline layouts
- Include comprehensive error checking for all Vulkan calls
