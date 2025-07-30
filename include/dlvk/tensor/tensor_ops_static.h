#pragma once

#include "tensor.h"

namespace dlvk {


class TensorOpsStatic {
public:

    static bool relu(const Tensor& input, Tensor& result);
    static bool sigmoid(const Tensor& input, Tensor& result);
    static bool tanh_activation(const Tensor& input, Tensor& result);
    static bool softmax(const Tensor& input, Tensor& result);
    

    static bool matrix_multiply(const Tensor& a, const Tensor& b, Tensor& result);
    

    static bool relu_backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
    static bool sigmoid_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input);
    static bool tanh_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input);
    

    static bool copy(const Tensor& source, Tensor& destination);
    static bool fill(Tensor& tensor, float value);
};

} // namespace dlvk
