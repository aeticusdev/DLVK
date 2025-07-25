#include "dlvk/data/dataset.h"

namespace dlvk {
namespace data {

TransformDataset::TransformDataset(std::shared_ptr<Dataset> dataset,
                                  Transform input_transform,
                                  Transform target_transform)
    : m_base_dataset(std::move(dataset))
    , m_input_transform(input_transform)
    , m_target_transform(target_transform) {
}

size_t TransformDataset::size() const {
    return m_base_dataset->size();
}

std::pair<Tensor, Tensor> TransformDataset::get_item(size_t index) const {
    auto [input, target] = m_base_dataset->get_item(index);
    
    if (m_input_transform) {
        input = m_input_transform(input);
    }
    
    if (m_target_transform) {
        target = m_target_transform(target);
    }
    
    return {input, target};
}

std::vector<size_t> TransformDataset::input_shape() const {
    return m_base_dataset->input_shape();
}

std::vector<size_t> TransformDataset::target_shape() const {
    return m_base_dataset->target_shape();
}

size_t TransformDataset::num_classes() const {
    return m_base_dataset->num_classes();
}

} // namespace data
} // namespace dlvk
