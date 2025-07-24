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
    src/layers/activation.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
    src/model/model.cpp
    src/model/callbacks.cpp
)

# Phase 5 Model APIs Demo (Simple)
add_executable(demo_phase5_simple
    demo_phase5_simple.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/layers/activation.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
    src/model/model.cpp
    src/model/callbacks.cpp
)

# Examples
add_executable(simple_training_demo
    examples/simple_training_demo.cpp
)

# Phase 5 Model APIs Demo
add_executable(demo_phase5_model_apis
    demo_phase5_model_apis.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/layers/activation.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
    src/model/model.cpp
    src/model/callbacks.cpp
)

# Phase 5 Foundation Demo
add_executable(demo_phase5_foundation
    demo_phase5_foundation.cpp
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

# Complete Pipeline Demo - Data to Model Integration
add_executable(complete_pipeline_demo
    examples/complete_pipeline_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/layers/activation.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
    src/model/model.cpp
    src/model/callbacks.cpp
    src/data/dataset.cpp
    src/data/mnist.cpp
    src/data/dataloader.cpp
)

# Phase 6.1 Data Infrastructure Demo
add_executable(data_pipeline_demo
    examples/data_pipeline_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/layers/activation.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
    src/model/model.cpp
    src/model/callbacks.cpp
    src/data/dataset.cpp
    src/data/mnist.cpp
    src/data/dataloader.cpp
)

# Phase 6.2 Complete Training Pipeline Demo
add_executable(training_pipeline_demo
    examples/training_pipeline_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/layers/activation.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
    src/model/model.cpp
    src/model/callbacks.cpp
    src/data/dataset.cpp
    src/data/mnist.cpp
    src/data/dataloader.cpp
    src/training/trainer.cpp
)

# Phase 6.2 Training Foundation Demo (Simplified)
add_executable(training_foundation_demo
    examples/training_foundation_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/compute/compute_pipeline.cpp
    src/data/dataset.cpp
    src/data/mnist.cpp
    src/data/dataloader.cpp
)

# Phase 6.3 Real Training Demo (Production Training Pipeline)
add_executable(real_training_demo
    examples/real_training_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/compute/compute_pipeline.cpp
    src/data/dataset.cpp
    src/data/mnist.cpp
    src/data/dataloader.cpp
    src/layers/dense_layer.cpp
    src/layers/activation.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/model/model.cpp
    src/loss/loss_functions.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/optimizers/optimizers.cpp
    src/training/trainer.cpp
)

# Phase 6.3 Real Production Demo (REAL Values Only!)
add_executable(real_production_demo
    examples/real_production_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
)

# Phase 6.3 Advanced Training Features Demo
add_executable(phase6_3_advanced_training_demo
    examples/phase6_3_advanced_training_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
)

# Real Model Training Demo - Actually train something!
add_executable(real_model_training_demo
    examples/real_model_training_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
)

# Real Model Persistence Demo - Uses actual implemented core library
add_executable(model_export_demo
    examples/model_export_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/core/model_persistence.cpp
)

target_include_directories(model_export_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(model_export_demo PRIVATE
    ${Vulkan_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

target_compile_features(model_export_demo PRIVATE cxx_std_17)

# Complete DLVK Framework Test - ALL Phases 1-6.3
add_executable(test_complete_dlvk_framework
    test_complete_dlvk_framework.cpp
)

target_include_directories(test_complete_dlvk_framework PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(test_complete_dlvk_framework
    dlvk  # Link against our static library
    ${Vulkan_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

target_compile_features(test_complete_dlvk_framework PRIVATE cxx_std_17)

# Simple ActivationLayer Test
add_executable(test_activation_only
    test_activation_only.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/activation.cpp
)

# Sequential Model Test
add_executable(test_sequential_model
    test_sequential_model.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/layers/activation.cpp
    src/optimizers/optimizers.cpp
    src/model/model.cpp
    src/loss/loss_functions.cpp
)

add_executable(test_tensor_ops_static
    test_tensor_ops_static.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/activation.cpp
)

# Add library target
add_library(dlvk STATIC
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/layers/activation.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/optimizers/optimizers.cpp
    src/loss/loss_functions.cpp
    src/loss/loss_ops_gpu.cpp
    src/model/model.cpp
    src/model/callbacks.cpp
    src/data/dataset.cpp
    src/data/dataloader.cpp
    src/data/mnist.cpp
    src/training/trainer.cpp
    src/training/model_persistence.cpp
    # Phase 6 Advanced Training Features
    src/training/mixed_precision.cpp
    src/training/regularization.cpp
    src/training/advanced_training.cpp
    # Phase 6 Optimization
    src/optimization/model_optimizer.cpp
    # Phase 6 Deployment
    src/deployment/multi_gpu_trainer.cpp
    src/deployment/inference_deployment.cpp
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

target_include_directories(demo_phase5_simple PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(demo_phase5_model_apis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(demo_phase5_foundation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(data_pipeline_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(training_pipeline_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(training_foundation_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(real_training_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(real_production_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(phase6_3_advanced_training_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(real_model_training_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(complete_pipeline_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(test_activation_only PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(test_sequential_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(test_activation_only PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_include_directories(test_tensor_ops_static PRIVATE
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

target_link_libraries(demo_phase5_simple PRIVATE
    dlvk
)

target_link_libraries(demo_phase5_model_apis PRIVATE
    dlvk
)

target_link_libraries(demo_phase5_foundation PRIVATE
    dlvk
)

target_link_libraries(test_activation_only PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(test_sequential_model PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(test_tensor_ops_static PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(data_pipeline_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(training_pipeline_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(training_foundation_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(real_training_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(real_production_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(phase6_3_advanced_training_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(real_model_training_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
)

target_link_libraries(complete_pipeline_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
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
add_dependencies(test_activation_only shaders)
add_dependencies(test_sequential_model shaders)
add_dependencies(test_tensor_ops_static shaders)
add_dependencies(real_production_demo shaders)
add_dependencies(phase6_3_advanced_training_demo shaders)

# Deployment and Optimization
add_executable(optimization_demo
    examples/optimization_demo.cpp
    src/core/vulkan_device.cpp
    src/core/vulkan_context.cpp
    src/core/buffer_manager.cpp
    src/core/pipeline_manager.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/tensor/tensor_ops_static.cpp
    src/compute/compute_pipeline.cpp
    src/layers/dense_layer.cpp
    src/layers/conv_layer.cpp
    src/layers/conv2d_layer.cpp
    src/layers/pooling_layers.cpp
    src/layers/batch_norm_layers.cpp
    src/layers/dropout_layer.cpp
    src/layers/activation.cpp
    src/optimizers/sgd.cpp
    src/optimizers/adam.cpp
    src/loss/loss_functions.cpp
    src/optimizers/optimizers.cpp
    src/model/model.cpp
    src/model/callbacks.cpp
    src/data/dataset.cpp
    src/data/mnist.cpp
    src/data/dataloader.cpp
    src/training/trainer.cpp
)

# Target includes for new optimization components
target_include_directories(optimization_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(optimization_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

target_compile_features(optimization_demo PRIVATE cxx_std_17)

# Phase 6 Complete Demo
add_executable(phase6_demo
    examples/phase6_demo.cpp
)

target_include_directories(phase6_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(phase6_demo PRIVATE
    dlvk
    ${Vulkan_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

target_compile_features(phase6_demo PRIVATE cxx_std_17)

# Testing
enable_testing()
add_subdirectory(tests)
