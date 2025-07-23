#include "dlvk/tensor/tensor.h"
#include "dlvk/core/vulkan_device.h"
#include <iostream>

int main() {
    try {
        std::cout << "Creating device..." << std::endl;
        auto device = std::make_shared<dlvk::VulkanDevice>();
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            return 1;
        }

        std::cout << "Creating tensor..." << std::endl;
        {
            // Create a simple tensor
            dlvk::Tensor tensor({10, 10}, dlvk::DataType::FLOAT32, device);
            std::cout << "Tensor created successfully!" << std::endl;
            std::cout << "Tensor size: " << tensor.size() << std::endl;
        }
        std::cout << "Tensor destroyed successfully!" << std::endl;

        std::cout << "Creating second tensor..." << std::endl;
        {
            // Create another tensor
            dlvk::Tensor tensor2({5, 5}, dlvk::DataType::FLOAT32, device);
            std::cout << "Second tensor created successfully!" << std::endl;
        }
        std::cout << "Second tensor destroyed successfully!" << std::endl;

        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
