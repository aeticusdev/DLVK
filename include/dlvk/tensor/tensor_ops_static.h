#pragma once

#include "tensor.h"

namespace dlvk {

// Static wrapper functions for global TensorOps instance
class TensorOpsStatic {
public:
    // Activation functions
    static bool relu(const Tensor& input, Tensor& result);
    static bool sigmoid(const Tensor& input, Tensor& result);
    static bool tanh_activation(const Tensor& input, Tensor& result);
    static bool softmax(const Tensor& input, Tensor& result);
    
    // Matrix operations
    static bool matrix_multiply(const Tensor& a, const Tensor& b, Tensor& result);
    
    // Backward pass functions
    static bool relu_backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
    static bool sigmoid_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input);
    static bool tanh_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input);
    
    // Utility functions
    static bool copy(const Tensor& source, Tensor& destination);
    static bool fill(Tensor& tensor, float value);
};

} // namespace dlvk
