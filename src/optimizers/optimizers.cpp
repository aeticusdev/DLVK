#include "dlvk/optimizers/optimizers.h"
#include "dlvk/layers/layer.h"

namespace dlvk {

SGD::SGD(float learning_rate) : m_learning_rate(learning_rate) {}

void SGD::update(Layer* layer) {
    if (layer) {
        layer->update_weights(m_learning_rate);
    }
}

} // namespace dlvk
