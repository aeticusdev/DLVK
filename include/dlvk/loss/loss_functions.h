#pragma once

#include <memory>
#include "dlvk/tensor/tensor.h"

namespace dlvk {

class LossFunction {
public:
    virtual ~LossFunction() = default;
    

    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& predictions, 
                                           const std::shared_ptr<Tensor>& targets) = 0;
    

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

class BinaryCrossEntropyLoss : public LossFunction {
private:
    float epsilon_;  // For numerical stability

public:
    BinaryCrossEntropyLoss(float epsilon = 1e-7f) : epsilon_(epsilon) {}
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& predictions, 
                                   const std::shared_ptr<Tensor>& targets) override;
    
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& predictions,
                                    const std::shared_ptr<Tensor>& targets) override;
};

} // namespace dlvk
