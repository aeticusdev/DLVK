#include "dlvk/tensor/tensor_ops_static.h"
#include "dlvk/tensor/tensor_ops.h"

namespace dlvk {

bool TensorOpsStatic::relu(const Tensor& input, Tensor& result) {
    auto* ops = TensorOps::instance();
    return ops ? ops->relu(input, result) : false;
}

bool TensorOpsStatic::sigmoid(const Tensor& input, Tensor& result) {
    auto* ops = TensorOps::instance();
    return ops ? ops->sigmoid(input, result) : false;
}

bool TensorOpsStatic::tanh_activation(const Tensor& input, Tensor& result) {
    auto* ops = TensorOps::instance();
    return ops ? ops->tanh_activation(input, result) : false;
}

bool TensorOpsStatic::softmax(const Tensor& input, Tensor& result) {
    auto* ops = TensorOps::instance();
    return ops ? ops->softmax(input, result) : false;
}

bool TensorOpsStatic::matrix_multiply(const Tensor& a, const Tensor& b, Tensor& result) {
    auto* ops = TensorOps::instance();
    return ops ? ops->matrix_multiply(a, b, result) : false;
}

bool TensorOpsStatic::relu_backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
    auto* ops = TensorOps::instance();
    return ops ? ops->relu_backward(input, grad_output, grad_input) : false;
}

bool TensorOpsStatic::sigmoid_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input) {
    auto* ops = TensorOps::instance();
    return ops ? ops->sigmoid_backward(output, grad_output, grad_input) : false;
}

bool TensorOpsStatic::tanh_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input) {
    auto* ops = TensorOps::instance();
    return ops ? ops->tanh_backward(output, grad_output, grad_input) : false;
}

bool TensorOpsStatic::copy(const Tensor& source, Tensor& destination) {
    auto* ops = TensorOps::instance();
    return ops ? ops->copy(source, destination) : false;
}

bool TensorOpsStatic::fill(Tensor& tensor, float value) {
    auto* ops = TensorOps::instance();
    return ops ? ops->fill(tensor, value) : false;
}

} // namespace dlvk
