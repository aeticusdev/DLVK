#pragma once

#include <memory>
#include "dlvk/tensor/tensor.h"

namespace dlvk {

class LossFunction {
public:
    virtual ~LossFunction() = default;
    
    // Compute the loss value
    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& predictions, 
                                           const std::shared_ptr<Tensor>& targets) = 0;
    
    // Compute gradients with respect to predictions
    virtual std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& predictions,
                                            const std::shared_ptr<Tensor>& targets) = 0;
};

class MeanSquaredError : public LossFunction {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& predictions, 
                                   const std::shared_ptr<Tensor>& targets) override;
    
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& predictions,
                                    const std::shared_ptr<Tensor>& targets) override;
};

class CrossEntropyLoss : public LossFunction {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& predictions, 
                                   const std::shared_ptr<Tensor>& targets) override;
    
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& predictions,
                                    const std::shared_ptr<Tensor>& targets) override;
};

} // namespace dlvk
