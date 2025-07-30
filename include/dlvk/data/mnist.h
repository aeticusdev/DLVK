#pragma once

#include "dataset.h"
#include <string>

namespace dlvk {
namespace data {

/**
 * @brief MNIST dataset loader
 * Loads the famous handwritten digit recognition dataset
 */
class MnistDataset : public Dataset {
private:
    std::vector<std::vector<float>> m_images;
    std::vector<int> m_labels;
    bool m_train;
    std::string m_root_dir;


    void load_images(const std::string& image_file);
    void load_labels(const std::string& label_file);
    uint32_t read_uint32_be(std::ifstream& file);
    void create_synthetic_data();

public:
    /**
     * @brief Constructor for MNIST dataset
     * @param root Directory to store/load MNIST data
     * @param train Whether to load training set (true) or test set (false)
     * @param download Whether to download data if not found
     */
    MnistDataset(const std::string& root, bool train = true, bool download = true);


    size_t size() const override;
    std::pair<Tensor, Tensor> get_item(size_t index) const override;
    std::vector<size_t> input_shape() const override;
    std::vector<size_t> target_shape() const override;
    size_t num_classes() const override;


    const std::vector<float>& get_image_data(size_t index) const;
    int get_label(size_t index) const;


    static constexpr size_t IMAGE_SIZE = 28;
    static constexpr size_t NUM_CLASSES = 10;
};

} // namespace data
} // namespace dlvk
