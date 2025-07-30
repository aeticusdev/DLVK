#pragma once





#include "dlvk/core/vulkan_device.h"


#include "dlvk/tensor/tensor.h"


#include "dlvk/layers/layer.h"


namespace dlvk {

    constexpr int VERSION_MAJOR = 0;
    constexpr int VERSION_MINOR = 1;
    constexpr int VERSION_PATCH = 0;
    

    using Shape = std::vector<size_t>;
    

    inline Shape make_shape(std::initializer_list<size_t> dims) {
        return Shape(dims);
    }
    

    inline std::string version_string() {
        return std::to_string(VERSION_MAJOR) + "." + 
               std::to_string(VERSION_MINOR) + "." + 
               std::to_string(VERSION_PATCH);
    }
}


#define DLVK_VERSION_STRING dlvk::version_string()
