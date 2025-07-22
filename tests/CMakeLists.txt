add_executable(test_tensor test_tensor.cpp)
add_executable(test_vulkan_device test_vulkan_device.cpp)

target_link_libraries(test_tensor PRIVATE dlvk)
target_link_libraries(test_vulkan_device PRIVATE dlvk)

target_include_directories(test_tensor PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_include_directories(test_vulkan_device PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME VulkanDeviceTest COMMAND test_vulkan_device)
