#include "dlvk/data/mnist.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <random>

namespace dlvk {
namespace data {

MnistDataset::MnistDataset(const std::string& root, bool train, bool download)
    : m_train(train), m_root_dir(root) {
    

    if (!std::filesystem::exists(m_root_dir)) {
        std::filesystem::create_directories(m_root_dir);
    }



    if (download) {
        std::cout << "Note: Automatic download not yet implemented." << std::endl;
        std::cout << "Please download MNIST data files to: " << m_root_dir << std::endl;
        std::cout << "Required files:" << std::endl;
        std::cout << "  - train-images-idx3-ubyte" << std::endl;
        std::cout << "  - train-labels-idx1-ubyte" << std::endl;
        std::cout << "  - t10k-images-idx3-ubyte" << std::endl;
        std::cout << "  - t10k-labels-idx1-ubyte" << std::endl;
    }


    std::string image_file, label_file;
    if (m_train) {
        image_file = m_root_dir + "/train-images-idx3-ubyte";
        label_file = m_root_dir + "/train-labels-idx1-ubyte";
    } else {
        image_file = m_root_dir + "/t10k-images-idx3-ubyte";
        label_file = m_root_dir + "/t10k-labels-idx1-ubyte";
    }


    if (!std::filesystem::exists(image_file) || !std::filesystem::exists(label_file)) {
        std::cout << "MNIST files not found. Creating synthetic data for demo..." << std::endl;
        create_synthetic_data();
        return;
    }

    try {
        load_images(image_file);
        load_labels(label_file);
        std::cout << "Loaded MNIST " << (m_train ? "training" : "test") 
                  << " set: " << m_images.size() << " samples" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error loading MNIST files: " << e.what() << std::endl;
        std::cout << "Creating synthetic data for demo..." << std::endl;
        create_synthetic_data();
    }
}

void MnistDataset::create_synthetic_data() {

    size_t num_samples = m_train ? 1000 : 200; // Smaller dataset for demo
    
    m_images.reserve(num_samples);
    m_labels.reserve(num_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dis(0, 9);
    
    for (size_t i = 0; i < num_samples; ++i) {

        std::vector<float> image(IMAGE_SIZE * IMAGE_SIZE);
        for (float& pixel : image) {
            pixel = dis(gen);
        }
        m_images.push_back(std::move(image));
        

        m_labels.push_back(label_dis(gen));
    }
    
    std::cout << "Created synthetic MNIST " << (m_train ? "training" : "test") 
              << " set: " << m_images.size() << " samples" << std::endl;
}

uint32_t MnistDataset::read_uint32_be(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));

    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x000000FF) << 24);
}

void MnistDataset::load_images(const std::string& image_file) {
    std::ifstream file(image_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MNIST image file: " + image_file);
    }


    uint32_t magic = read_uint32_be(file);
    if (magic != 0x00000803) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    uint32_t num_images = read_uint32_be(file);
    uint32_t rows = read_uint32_be(file);
    uint32_t cols = read_uint32_be(file);

    if (rows != IMAGE_SIZE || cols != IMAGE_SIZE) {
        throw std::runtime_error("Unexpected MNIST image dimensions");
    }


    m_images.reserve(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
        std::vector<float> image(IMAGE_SIZE * IMAGE_SIZE);
        for (size_t j = 0; j < IMAGE_SIZE * IMAGE_SIZE; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = static_cast<float>(pixel) / 255.0f; // Normalize to [0, 1]
        }
        m_images.push_back(std::move(image));
    }
}

void MnistDataset::load_labels(const std::string& label_file) {
    std::ifstream file(label_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MNIST label file: " + label_file);
    }


    uint32_t magic = read_uint32_be(file);
    if (magic != 0x00000801) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t num_labels = read_uint32_be(file);


    m_labels.reserve(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        m_labels.push_back(static_cast<int>(label));
    }
}

size_t MnistDataset::size() const {
    return m_images.size();
}

std::pair<Tensor, Tensor> MnistDataset::get_item(size_t index) const {
    if (index >= m_images.size()) {
        throw std::out_of_range("Index out of range");
    }




    throw std::runtime_error("get_item() requires device context - use with DataLoader instead");
}

std::vector<size_t> MnistDataset::input_shape() const {
    return {1, IMAGE_SIZE, IMAGE_SIZE};
}

std::vector<size_t> MnistDataset::target_shape() const {
    return {NUM_CLASSES};
}

size_t MnistDataset::num_classes() const {
    return NUM_CLASSES;
}

const std::vector<float>& MnistDataset::get_image_data(size_t index) const {
    if (index >= m_images.size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_images[index];
}

int MnistDataset::get_label(size_t index) const {
    if (index >= m_labels.size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_labels[index];
}

} // namespace data
} // namespace dlvk
