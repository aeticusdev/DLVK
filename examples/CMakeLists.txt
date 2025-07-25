# DLVK Examples CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Common source files
set(DLVK_CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/vulkan_device.cpp
    ${CMAKE_SOURCE_DIR}/src/tensor/tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/tensor/tensor_ops.cpp
    ${CMAKE_SOURCE_DIR}/src/training/model_persistence.cpp
)

# Real Model Persistence Demo
add_executable(real_model_persistence_demo
    real_model_persistence_demo.cpp
    ${DLVK_CORE_SOURCES}
)

target_link_libraries(real_model_persistence_demo
    Vulkan::Vulkan
    ${CMAKE_DL_LIBS}
)

target_compile_features(real_model_persistence_demo PRIVATE cxx_std_17)

# Enhanced Model Export Demo
add_executable(enhanced_model_export
    enhanced_model_export.cpp
    ${DLVK_CORE_SOURCES}
)

target_link_libraries(enhanced_model_export
    Vulkan::Vulkan
    ${CMAKE_DL_LIBS}
)

target_compile_features(enhanced_model_export PRIVATE cxx_std_17)

# Final Training Demo
add_executable(final_training_demo
    final_training_demo.cpp
    ${DLVK_CORE_SOURCES}
)

target_link_libraries(final_training_demo
    Vulkan::Vulkan
    ${CMAKE_DL_LIBS}
)

target_compile_features(final_training_demo PRIVATE cxx_std_17)

# Set output directory
set_target_properties(real_model_persistence_demo enhanced_model_export final_training_demo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

# Add custom target to build all examples
add_custom_target(build_examples
    DEPENDS real_model_persistence_demo enhanced_model_export final_training_demo
)

message(STATUS "DLVK Examples configured with real persistence system")
