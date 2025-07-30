#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <dlvk/model/model.h>
#include <dlvk/tensor/tensor.h>


#ifdef DLVK_HDF5_SUPPORT
#include <hdf5.h>
#else

typedef int hid_t;
#endif

namespace dlvk {
namespace export_formats {

/**
 * @brief Supported model export formats
 */
enum class ModelFormat {
    DLVK,
    ONNX,
    GGUF,
    HDF5,
    SAFETENSORS,
    TENSORFLOW_SAVEDMODEL
};

/**
 * @brief Model metadata for export
 */
struct ModelMetadata {
    std::string name;
    std::string architecture;
    std::string version;
    std::string framework;
    std::map<std::string, std::string> custom_metadata;
    

    size_t vocab_size = 0;
    size_t embedding_dim = 0;
    size_t hidden_size = 0;
    size_t sequence_length = 0;
    size_t num_layers = 0;
};

/**
 * @brief Tensor information for export
 */
struct TensorInfo {
    std::string name;
    std::vector<size_t> shape;
    DataType dtype;
    std::shared_ptr<Tensor> data;
    size_t offset = 0;
    size_t size_bytes = 0;
};

/**
 * @brief Base class for model exporters
 */
class ModelExporter {
public:
    virtual ~ModelExporter() = default;
    
    virtual bool export_model(
        const std::shared_ptr<Model>& model,
        const std::string& filename,
        const ModelMetadata& metadata
    ) = 0;
    
    virtual bool import_model(
        const std::string& filename,
        std::shared_ptr<Model>& model,
        ModelMetadata& metadata
    ) = 0;
    
protected:
    std::vector<TensorInfo> extract_tensors(const std::shared_ptr<Model>& model);
    void write_binary_data(std::ofstream& file, const std::vector<TensorInfo>& tensors);
    void read_binary_data(std::ifstream& file, std::vector<TensorInfo>& tensors);
};

/**
 * @brief Native DLVK format exporter
 */
class DLVKExporter : public ModelExporter {
public:
    bool export_model(
        const std::shared_ptr<Model>& model,
        const std::string& filename,
        const ModelMetadata& metadata
    ) override;
    
    bool import_model(
        const std::string& filename,
        std::shared_ptr<Model>& model,
        ModelMetadata& metadata
    ) override;

private:
    void write_dlvk_header(std::ofstream& file, const ModelMetadata& metadata);
    void write_dlvk_tensors(std::ofstream& file, const std::vector<TensorInfo>& tensors);
};

/**
 * @brief ONNX format exporter
 */
class ONNXExporter : public ModelExporter {
public:
    bool export_model(
        const std::shared_ptr<Model>& model,
        const std::string& filename,
        const ModelMetadata& metadata
    ) override;
    
    bool import_model(
        const std::string& filename,
        std::shared_ptr<Model>& model,
        ModelMetadata& metadata
    ) override;

private:
    void write_onnx_header(std::ofstream& file, const ModelMetadata& metadata);
    void write_onnx_graph(std::ofstream& file, const std::vector<TensorInfo>& tensors, const ModelMetadata& metadata);
    void write_onnx_weights(std::ofstream& file, const std::vector<TensorInfo>& tensors);
};

/**
 * @brief GGUF format exporter (llama.cpp compatible)
 */
class GGUFExporter : public ModelExporter {
public:
    bool export_model(
        const std::shared_ptr<Model>& model,
        const std::string& filename,
        const ModelMetadata& metadata
    ) override;
    
    bool import_model(
        const std::string& filename,
        std::shared_ptr<Model>& model,
        ModelMetadata& metadata
    ) override;

private:
    void write_gguf_header(std::ofstream& file, const ModelMetadata& metadata, const std::vector<TensorInfo>& tensors);
    void write_gguf_metadata(std::ofstream& file, const ModelMetadata& metadata);
    void write_gguf_tensor_info(std::ofstream& file, const std::vector<TensorInfo>& tensors);
    void write_gguf_tensor_data(std::ofstream& file, const std::vector<TensorInfo>& tensors);
};

/**
 * @brief HDF5 format exporter (Keras/TensorFlow compatible)
 */
class HDF5Exporter : public ModelExporter {
public:
    bool export_model(
        const std::shared_ptr<Model>& model,
        const std::string& filename,
        const ModelMetadata& metadata
    ) override;
    
    bool import_model(
        const std::string& filename,
        std::shared_ptr<Model>& model,
        ModelMetadata& metadata
    ) override;

private:

    void write_hdf5_attribute(hid_t group_id, const std::string& name, const std::string& value);
    void write_hdf5_attribute(hid_t group_id, const std::string& name, int value);
    void write_hdf5_tensor(hid_t group_id, const TensorInfo& tensor);
    std::string read_hdf5_string_attribute(hid_t group_id, const std::string& name);
    int read_hdf5_int_attribute(hid_t group_id, const std::string& name);
};

/**
 * @brief SafeTensors format exporter (Hugging Face compatible)
 */
class SafeTensorsExporter : public ModelExporter {
public:
    bool export_model(
        const std::shared_ptr<Model>& model,
        const std::string& filename,
        const ModelMetadata& metadata
    ) override;
    
    bool import_model(
        const std::string& filename,
        std::shared_ptr<Model>& model,
        ModelMetadata& metadata
    ) override;

private:
    std::string create_safetensors_header(const std::vector<TensorInfo>& tensors, const ModelMetadata& metadata);
    void write_safetensors_data(std::ofstream& file, const std::vector<TensorInfo>& tensors);
};

/**
 * @brief Model export manager - main interface
 */
class ModelExportManager {
private:
    std::map<ModelFormat, std::unique_ptr<ModelExporter>> m_exporters;
    
public:
    ModelExportManager();
    ~ModelExportManager() = default;
    

    bool export_model(
        const std::shared_ptr<Model>& model,
        const std::string& filename,
        ModelFormat format,
        const ModelMetadata& metadata = {}
    );
    

    bool import_model(
        const std::string& filename,
        ModelFormat format,
        std::shared_ptr<Model>& model,
        ModelMetadata& metadata
    );
    

    ModelFormat detect_format(const std::string& filename);
    

    bool export_all_formats(
        const std::shared_ptr<Model>& model,
        const std::string& base_filename,
        const ModelMetadata& metadata = {}
    );
    

    std::vector<ModelFormat> get_supported_formats() const;
    

    std::string get_extension(ModelFormat format) const;
};

/**
 * @brief Factory function for creating export manager
 */
std::unique_ptr<ModelExportManager> create_export_manager();

} // namespace export_formats
} // namespace dlvk
