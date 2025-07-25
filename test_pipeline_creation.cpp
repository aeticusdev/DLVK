#include <iostream>
#include <memory>
#include "dlvk/core/vulkan_device.h"
#include "dlvk/loss/loss_ops_gpu.h"

using namespace dlvk;

int main() {
    std::cout << "=== Testing Loss Pipeline Creation ===" << std::endl;
    
    auto device = std::make_shared<VulkanDevice>();
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return -1;
    }
    
    std::cout << "✓ GPU: " << device->get_device_name() << std::endl;
    
    auto loss_ops = std::make_unique<LossOpsGPU>(device);
    std::cout << "\n--- Initializing LossOpsGPU ---" << std::endl;
    
    if (loss_ops->initialize()) {
        std::cout << "✓ LossOpsGPU initialized successfully" << std::endl;
    } else {
        std::cout << "✗ LossOpsGPU initialization failed" << std::endl;
    }
    
    return 0;
}
