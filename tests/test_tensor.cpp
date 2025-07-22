#include <iostream>
#include "dlvk/tensor/tensor.h"
#include "dlvk/core/vulkan_device.h"

int main() {
    std::cout << "Testing Tensor operations...\n";
    
    try {
        auto device = std::make_shared<dlvk::VulkanDevice>();
        if (!device->initialize()) {
            std::cout << "Failed to initialize Vulkan device\n";
            return -1;
        }
        
        // Test tensor creation
        std::vector<size_t> shape = {4, 4};
        auto tensor = std::make_shared<dlvk::Tensor>(shape, dlvk::DataType::FLOAT32, device);
        
        std::cout << "✓ Tensor created successfully\n";
        std::cout << "✓ Shape: [" << shape[0] << ", " << shape[1] << "]\n";
        std::cout << "✓ Size: " << tensor->size() << " elements\n";
        
        // Test data upload/download
        std::vector<float> data(16);
        for (int i = 0; i < 16; ++i) {
            data[i] = static_cast<float>(i);
        }
        
        tensor->upload_data(data.data());
        
        std::vector<float> retrieved_data(16);
        tensor->download_data(retrieved_data.data());
        
        bool data_matches = true;
        for (int i = 0; i < 16; ++i) {
            if (data[i] != retrieved_data[i]) {
                data_matches = false;
                break;
            }
        }
        
        if (data_matches) {
            std::cout << "✓ Data upload/download test passed\n";
        } else {
            std::cout << "✗ Data upload/download test failed\n";
            return -1;
        }
        
        std::cout << "All tensor tests passed!\n";
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
