cmake_minimum_required(VERSION 3.20)
project(DLVK VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Vulkan REQUIRED)
# Optional packages for examples and advanced features
find_package(glfw3 3.3 QUIET)
find_package(glm QUIET)

# Add executable targets
add_executable(dlvk_demo
    src/main.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
)

# Phase 4.3 CNN GPU Acceleration Demo
add_executable(demo_phase4_3_cnn_gpu
    demo_phase4_3_cnn_gpu.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
)

# Gradient Clipping Test
add_executable(test_gradient_clipping
    test_gradient_clipping.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
)

# Add library target
add_library(dlvk STATIC
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
)

# Include directories
target_include_directories(dlvk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(dlvk_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(demo_phase4_3_cnn_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(test_gradient_clipping PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(dlvk PUBLIC
    ${Vulkan_LIBRARIES}
)

if(glfw3_FOUND)
    target_link_libraries(dlvk PUBLIC glfw)
endif()

if(glm_FOUND)
    target_link_libraries(dlvk PUBLIC glm::glm)
endif()

target_link_libraries(dlvk_demo PRIVATE
    dlvk
)

target_link_libraries(demo_phase4_3_cnn_gpu PRIVATE
    dlvk
)

target_link_libraries(test_gradient_clipping PRIVATE
    dlvk
)

# Compiler-specific options
if(MSVC)
    target_compile_options(dlvk PRIVATE /W4)
    target_compile_options(dlvk_demo PRIVATE /W4)
    target_compile_options(demo_phase4_3_cnn_gpu PRIVATE /W4)
else()
    target_compile_options(dlvk PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(dlvk_demo PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(demo_phase4_3_cnn_gpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Shader compilation
find_program(GLSL_VALIDATOR glslangValidator HINTS
    ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
    /usr/bin
    /usr/local/bin
    ${VULKAN_SDK_PATH}/Bin
    ${VULKAN_SDK_PATH}/Bin32
    $ENV{VULKAN_SDK}/Bin/
    $ENV{VULKAN_SDK}/Bin32/
)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "shaders/*.comp"
    "shaders/*.vert"
    "shaders/*.frag"
)

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL}
    )
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    shaders
    DEPENDS ${SPIRV_BINARY_FILES}
)

add_dependencies(dlvk shaders)
add_dependencies(dlvk_demo shaders)
add_dependencies(demo_phase4_3_cnn_gpu shaders)
add_dependencies(test_gradient_clipping shaders)

# Testing
enable_testing()
add_subdirectory(tests)
