#pragma once

#include <cmath>

namespace dlvk {

class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(int step, float base_lr) = 0;
};

class StepLRScheduler : public LRScheduler {
private:
    int step_size_;
    float gamma_;

public:
    StepLRScheduler(int step_size, float gamma = 0.1f)
        : step_size_(step_size), gamma_(gamma) {}
    
    float get_lr(int step, float base_lr) override {
        int num_reductions = step / step_size_;
        return base_lr * std::pow(gamma_, num_reductions);
    }
};

class ExponentialLRScheduler : public LRScheduler {
private:
    float gamma_;

public:
    ExponentialLRScheduler(float gamma = 0.9f) : gamma_(gamma) {}
    
    float get_lr(int step, float base_lr) override {
        return base_lr * std::pow(gamma_, step);
    }
};

class CosineAnnealingLRScheduler : public LRScheduler {
private:
    int T_max_;
    float eta_min_;

public:
    CosineAnnealingLRScheduler(int T_max, float eta_min = 0.0f)
        : T_max_(T_max), eta_min_(eta_min) {}
    
    float get_lr(int step, float base_lr) override {
        float cos_inner = M_PI * (step % T_max_) / T_max_;
        return eta_min_ + (base_lr - eta_min_) * (1 + std::cos(cos_inner)) / 2;
    }
};

class LinearLRScheduler : public LRScheduler {
private:
    int total_steps_;
    float end_factor_;

public:
    LinearLRScheduler(int total_steps, float end_factor = 0.0f)
        : total_steps_(total_steps), end_factor_(end_factor) {}
    
    float get_lr(int step, float base_lr) override {
        if (step >= total_steps_) {
            return base_lr * end_factor_;
        }
        float factor = 1.0f + (end_factor_ - 1.0f) * step / total_steps_;
        return base_lr * factor;
    }
};

} // namespace dlvk
