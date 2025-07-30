# Contributing to DLVK

Thank you for your interest in contributing to DLVK! This document provides guidelines for contributing to the Deep Learning Vulkan Kit project.

## 🚀 Getting Started

### Prerequisites
- **C++20 compatible compiler** (GCC 10+, Clang 12+, MSVC 2019+)
- **CMake 3.20+**
- **Vulkan SDK 1.3+**
- **Git** for version control

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/DLVK.git
   cd DLVK
   ```

2. **Set up the build environment**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   cmake --build .
   ```

3. **Run tests to verify setup**
   ```bash
   ./test_tensor
   ./test_vulkan_device
   ./test_phase4_2_features
   ```

## 📋 Contribution Guidelines

### Code Style

- **C++20 Modern Features**: Use smart pointers, RAII, constexpr where appropriate
- **Naming Convention**: 
  - Classes: `PascalCase` (e.g., `VulkanDevice`, `BatchNorm1DLayer`)
  - Functions/Variables: `snake_case` (e.g., `forward_pass`, `learning_rate`)
  - Private members: `trailing_underscore_` (e.g., `device_`, `weights_`)
- **Include Guards**: Use `#pragma once` in headers
- **Documentation**: Document public APIs with clear comments

### File Organization

```
include/dlvk/           # Public headers
├── core/              # Core functionality (VulkanDevice, Tensor)
├── layers/            # Neural network layers
├── loss/              # Loss functions
├── optimizers/        # Optimizers and schedulers
└── compute/           # GPU compute pipelines

src/                   # Implementation files
├── core/
├── layers/
├── loss/
├── optimizers/
└── compute/

examples/              # Example code and tutorials
tests/                 # Test files
shaders/               # GLSL compute shaders
```

### Commit Message Format

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `build`: Build system changes

**Examples:**
```
feat(layers): add LSTM layer implementation
fix(tensor): resolve memory leak in GPU buffer allocation
docs(readme): update installation instructions
test(conv2d): add comprehensive convolution tests
```

## 🧩 Types of Contributions

### 1. **Bug Fixes**
- Check existing issues before creating new ones
- Include minimal reproduction code
- Add regression tests when possible

### 2. **New Features**
- Discuss major features in GitHub Issues first
- Follow the existing architecture patterns
- Include comprehensive tests
- Update documentation

### 3. **Performance Improvements**
- Include benchmarks showing improvement
- Ensure no functionality regression
- Document performance characteristics

### 4. **Documentation**
- API documentation in code comments
- Tutorial examples in `examples/`
- Architecture documentation in `docs/`

## 🧪 Testing Guidelines

### Test Requirements
- **Unit Tests**: For individual components
- **Integration Tests**: For feature interactions
- **Performance Tests**: For optimization validation

### Running Tests
```bash
# Core functionality
./test_tensor
./test_vulkan_device

# Advanced features
./test_phase4_2_features

# Build and run all tests
cmake --build . --target test
```

### Writing Tests
- Test both success and failure cases
- Include edge cases and boundary conditions
- Use descriptive test names
- Add performance benchmarks for GPU operations

## 🏗️ Architecture Guidelines

### Adding New Layers
1. **Header**: Define interface in `include/dlvk/layers/`
2. **Implementation**: Add implementation in `src/layers/`
3. **Tests**: Create comprehensive tests
4. **Documentation**: Update API docs and examples

Example layer structure:
```cpp
class CustomLayer : public Layer {
public:
    CustomLayer(VulkanDevice& device, /* parameters */);
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& grad_output) override;
    void update_weights(float learning_rate) override;
    
private:
    VulkanDevice& device_;

};
```

### GPU Compute Shaders
- Write efficient GLSL compute shaders in `shaders/`
- Use proper memory access patterns
- Include comprehensive error checking
- Add performance profiling

### Memory Management
- Use RAII for all resources
- Prefer smart pointers over raw pointers
- Ensure proper GPU memory cleanup
- Handle Vulkan object lifecycles correctly

## 🐛 Issue Reporting

### Bug Reports
Include:
- **Environment**: OS, GPU, Vulkan version, compiler
- **Code**: Minimal reproduction case
- **Expected vs Actual**: Clear description of the problem
- **Logs**: Relevant error messages or debug output

### Feature Requests
Include:
- **Use Case**: Why this feature is needed
- **API Design**: Proposed interface
- **Implementation**: High-level approach
- **Alternatives**: Other solutions considered

## 📝 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Test Locally**
   ```bash
   cmake --build build
   cd build && ctest
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat(scope): your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Clear title and description
   - Reference related issues
   - Include test results
   - Add screenshots for UI changes

### PR Review Criteria
- ✅ Code compiles without warnings
- ✅ All tests pass
- ✅ Code follows style guidelines
- ✅ Documentation is updated
- ✅ Performance impact is acceptable

## 🎯 Development Priorities

### Current Focus (Phase 4.3)
- **GPU Acceleration**: Conv2D and pooling compute shaders
- **Model APIs**: Sequential and Functional model builders
- **Training Infrastructure**: Checkpointing and metrics

### Future Priorities
- **Advanced Architectures**: ResNet, Attention mechanisms
- **Multi-GPU Support**: Distributed training
- **Deployment**: Inference optimization

## 💬 Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Code Reviews**: Collaborative improvement

## 📄 License

By contributing to DLVK, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to DLVK! Together we're building the future of high-performance deep learning.**
