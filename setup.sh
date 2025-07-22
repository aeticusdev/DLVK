#!/bin/bash

echo "DLVK - Vulkan Machine Learning Framework Setup"
echo "=============================================="
echo

# Check for Vulkan support
echo "Checking system requirements..."

# Check for Vulkan SDK
if command -v glslangValidator &> /dev/null; then
    echo "✓ GLSL validator found"
else
    echo "✗ GLSL validator not found. Please install Vulkan SDK:"
    echo "  Ubuntu/Debian: sudo apt install vulkan-tools vulkan-validationlayers-dev spirv-tools"
    echo "  Arch: sudo pacman -S vulkan-tools vulkan-validation-layers spirv-tools"
    echo "  Or download from: https://vulkan.lunarg.com/"
fi

# Check for Vulkan runtime
if command -v vulkaninfo &> /dev/null; then
    echo "✓ Vulkan runtime found"
    echo "Available Vulkan devices:"
    vulkaninfo --summary 2>/dev/null | grep -A5 "VkPhysicalDevice" || echo "  Run 'vulkaninfo' for detailed information"
else
    echo "✗ Vulkan runtime not found. Install vulkan-tools package"
fi

# Check for CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    echo "✓ CMake found (version $CMAKE_VERSION)"
else
    echo "✗ CMake not found. Please install CMake 3.20 or later"
fi

# Check for compiler
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    echo "✓ GCC found (version $GCC_VERSION)"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    echo "✓ Clang found (version $CLANG_VERSION)"
else
    echo "✗ C++ compiler not found. Please install g++ or clang++"
fi

echo
echo "Setting up build environment..."

# Create build directory
mkdir -p build
cd build

echo "Configuring project with CMake..."
if cmake .. -DCMAKE_BUILD_TYPE=Debug; then
    echo "✓ CMake configuration successful"
    
    echo "Building project..."
    if make -j$(nproc) 2>/dev/null || make -j4 2>/dev/null || make; then
        echo "✓ Build successful"
        
        echo
        echo "Running demo..."
        if [ -f "./dlvk_demo" ]; then
            ./dlvk_demo
        else
            echo "Demo executable not found. Build may have issues."
        fi
    else
        echo "✗ Build failed. Check error messages above."
        echo
        echo "Common solutions:"
        echo "1. Install missing dependencies:"
        echo "   sudo apt install libglfw3-dev libglm-dev libvulkan-dev"
        echo "2. Make sure Vulkan SDK is properly installed"
        echo "3. Check compiler supports C++20"
    fi
else
    echo "✗ CMake configuration failed"
    echo
    echo "Trying to install dependencies..."
    echo "For Ubuntu/Debian:"
    echo "  sudo apt update"
    echo "  sudo apt install cmake build-essential libglfw3-dev libglm-dev libvulkan-dev vulkan-tools"
    echo
    echo "For Arch Linux:"
    echo "  sudo pacman -S cmake gcc glfw glm vulkan-headers vulkan-tools"
fi

cd ..

echo
echo "Setup completed. Check output above for any issues."
echo "To manually build:"
echo "  cd build && make -j\$(nproc)"
echo "To run demo:"
echo "  cd build && ./dlvk_demo"
