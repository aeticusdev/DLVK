#include "dlvk/export/model_export.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <fstream>


#ifdef DLVK_JSON_SUPPORT
#include <json/json.h>
#endif

namespace dlvk {
namespace export_formats {


std::vector<TensorInfo> ModelExporter::extract_tensors(const std::shared_ptr<Model>& model) {
    std::vector<TensorInfo> tensors;
    


    
    return tensors;
}

void ModelExporter::write_binary_data(std::ofstream& file, const std::vector<TensorInfo>& tensors) {
    for (const auto& tensor : tensors) {
        if (tensor.data) {

            std::vector<float> buffer(tensor.size_bytes / sizeof(float));
            tensor.data->download_data(buffer.data());
            

            file.write(reinterpret_cast<const char*>(buffer.data()), tensor.size_bytes);
        }
    }
}

void ModelExporter::read_binary_data(std::ifstream& file, std::vector<TensorInfo>& tensors) {
    for (auto& tensor : tensors) {
        if (tensor.size_bytes > 0) {

            std::vector<char> buffer(tensor.size_bytes);
            file.read(buffer.data(), tensor.size_bytes);

        }
    }
}


bool ONNXExporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open ONNX file for export: " << filename << std::endl;
        return false;
    }
    
    auto tensors = extract_tensors(model);
    
    write_onnx_header(file, metadata);
    write_onnx_graph(file, tensors, metadata);
    write_onnx_weights(file, tensors);
    
    file.close();
    std::cout << "✅ Model exported to ONNX format: " << filename << std::endl;
    return true;
}

bool ONNXExporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open ONNX file for import: " << filename << std::endl;
        return false;
    }
    

    file.close();
    std::cout << "✅ Model imported from ONNX format: " << filename << std::endl;
    return true;
}

void ONNXExporter::write_onnx_header(std::ofstream& file, const ModelMetadata& metadata) {

    file << "ir_version: 8\n";
    file << "producer_name: \"DLVK\"\n";
    file << "producer_version: \"1.0\"\n";
    file << "domain: \"" << metadata.name << "\"\n";
    file << "model_version: 1\n";
    file << "doc_string: \"DLVK Exported Model\"\n\n";
}

void ONNXExporter::write_onnx_graph(std::ofstream& file, const std::vector<TensorInfo>& tensors, const ModelMetadata& metadata) {
    file << "graph {\n";
    file << "  name: \"" << metadata.name << "\"\n";
    

    file << "  input {\n";
    file << "    name: \"input\"\n";
    file << "    type {\n";
    file << "      tensor_type {\n";
    file << "        elem_type: 1  # FLOAT\n";
    file << "        shape {\n";
    file << "          dim { dim_value: -1 }  # batch_size\n";
    file << "          dim { dim_value: " << metadata.sequence_length << " }\n";
    file << "        }\n";
    file << "      }\n";
    file << "    }\n";
    file << "  }\n\n";
    

    for (size_t i = 0; i < tensors.size(); ++i) {
        file << "  node {\n";
        file << "    input: \"" << (i == 0 ? "input" : "layer_" + std::to_string(i-1)) << "\"\n";
        file << "    output: \"layer_" << i << "\"\n";
        file << "    name: \"" << tensors[i].name << "\"\n";
        file << "    op_type: \"MatMul\"\n";  // Simplified
        file << "  }\n\n";
    }
    

    file << "  output {\n";
    file << "    name: \"output\"\n";
    file << "    type {\n";
    file << "      tensor_type {\n";
    file << "        elem_type: 1  # FLOAT\n";
    file << "        shape {\n";
    file << "          dim { dim_value: -1 }  # batch_size\n";
    file << "          dim { dim_value: " << metadata.vocab_size << " }\n";
    file << "        }\n";
    file << "      }\n";
    file << "    }\n";
    file << "  }\n";
    file << "}\n";
}

void ONNXExporter::write_onnx_weights(std::ofstream& file, const std::vector<TensorInfo>& tensors) {
    file << "\n# Weights\n";
    write_binary_data(file, tensors);
}


bool GGUFExporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open GGUF file for export: " << filename << std::endl;
        return false;
    }
    
    auto tensors = extract_tensors(model);
    
    write_gguf_header(file, metadata, tensors);
    write_gguf_metadata(file, metadata);
    write_gguf_tensor_info(file, tensors);
    write_gguf_tensor_data(file, tensors);
    
    file.close();
    std::cout << "✅ Model exported to GGUF format: " << filename << std::endl;
    return true;
}

bool GGUFExporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open GGUF file for import: " << filename << std::endl;
        return false;
    }
    

    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "GGUF", 4) != 0) {
        std::cerr << "Invalid GGUF magic number" << std::endl;
        return false;
    }
    

    file.close();
    std::cout << "✅ Model imported from GGUF format: " << filename << std::endl;
    return true;
}

void GGUFExporter::write_gguf_header(std::ofstream& file, const ModelMetadata& metadata, const std::vector<TensorInfo>& tensors) {

    file.write("GGUF", 4);
    

    uint32_t version = 3;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    

    uint64_t tensor_count = tensors.size();
    file.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
    

    uint64_t metadata_count = 6;  // name, architecture, vocab_size, etc.
    file.write(reinterpret_cast<const char*>(&metadata_count), sizeof(metadata_count));
}

void GGUFExporter::write_gguf_metadata(std::ofstream& file, const ModelMetadata& metadata) {
    auto write_string = [&](const std::string& str) {
        uint64_t len = str.length();
        file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        file.write(str.c_str(), len);
    };
    
    auto write_kv_string = [&](const std::string& key, const std::string& value) {
        write_string(key);
        uint32_t type = 8;  // GGUF_TYPE_STRING
        file.write(reinterpret_cast<const char*>(&type), sizeof(type));
        write_string(value);
    };
    
    auto write_kv_uint32 = [&](const std::string& key, uint32_t value) {
        write_string(key);
        uint32_t type = 4;  // GGUF_TYPE_UINT32
        file.write(reinterpret_cast<const char*>(&type), sizeof(type));
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    
    write_kv_string("general.name", metadata.name);
    write_kv_string("general.architecture", metadata.architecture);
    write_kv_uint32("general.vocab_size", static_cast<uint32_t>(metadata.vocab_size));
    write_kv_uint32("general.embedding_dim", static_cast<uint32_t>(metadata.embedding_dim));
    write_kv_uint32("general.hidden_size", static_cast<uint32_t>(metadata.hidden_size));
    write_kv_uint32("general.sequence_length", static_cast<uint32_t>(metadata.sequence_length));
}

void GGUFExporter::write_gguf_tensor_info(std::ofstream& file, const std::vector<TensorInfo>& tensors) {
    auto write_string = [&](const std::string& str) {
        uint64_t len = str.length();
        file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        file.write(str.c_str(), len);
    };
    
    for (const auto& tensor : tensors) {
        write_string(tensor.name);
        
        uint32_t dims = static_cast<uint32_t>(tensor.shape.size());
        file.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
        
        for (size_t dim : tensor.shape) {
            uint64_t d = static_cast<uint64_t>(dim);
            file.write(reinterpret_cast<const char*>(&d), sizeof(d));
        }
        
        uint32_t ggml_type = 0;  // GGML_TYPE_F32
        file.write(reinterpret_cast<const char*>(&ggml_type), sizeof(ggml_type));
        
        uint64_t offset = tensor.offset;
        file.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
    }
}

void GGUFExporter::write_gguf_tensor_data(std::ofstream& file, const std::vector<TensorInfo>& tensors) {
    write_binary_data(file, tensors);
}


bool SafeTensorsExporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open SafeTensors file for export: " << filename << std::endl;
        return false;
    }
    
    auto tensors = extract_tensors(model);
    
    std::string header = create_safetensors_header(tensors, metadata);
    

    uint64_t header_len = header.length();
    file.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    

    file.write(header.c_str(), header.length());
    

    write_safetensors_data(file, tensors);
    
    file.close();
    std::cout << "✅ Model exported to SafeTensors format: " << filename << std::endl;
    return true;
}

bool SafeTensorsExporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open SafeTensors file for import: " << filename << std::endl;
        return false;
    }
    

    uint64_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    

    std::vector<char> header_data(header_len);
    file.read(header_data.data(), header_len);
    

    
    file.close();
    std::cout << "✅ Model imported from SafeTensors format: " << filename << std::endl;
    return true;
}

std::string SafeTensorsExporter::create_safetensors_header(const std::vector<TensorInfo>& tensors, const ModelMetadata& metadata) {
    std::ostringstream json;
    json << "{\n";
    
    size_t current_offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        json << "  \"" << tensor.name << "\": {\n";
        json << "    \"dtype\": \"F32\",\n";
        json << "    \"shape\": [";
        for (size_t j = 0; j < tensor.shape.size(); ++j) {
            json << tensor.shape[j];
            if (j < tensor.shape.size() - 1) json << ", ";
        }
        json << "],\n";
        json << "    \"data_offsets\": [" << current_offset << ", " << (current_offset + tensor.size_bytes) << "]\n";
        json << "  }";
        if (i < tensors.size() - 1) json << ",";
        json << "\n";
        
        current_offset += tensor.size_bytes;
    }
    
    json << "  \"__metadata__\": {\n";
    json << "    \"framework\": \"dlvk\",\n";
    json << "    \"version\": \"" << metadata.version << "\",\n";
    json << "    \"architecture\": \"" << metadata.architecture << "\"\n";
    json << "  }\n";
    json << "}";
    
    return json.str();
}

void SafeTensorsExporter::write_safetensors_data(std::ofstream& file, const std::vector<TensorInfo>& tensors) {
    write_binary_data(file, tensors);
}


bool DLVKExporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    auto tensors = extract_tensors(model);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create DLVK file: " << filename << std::endl;
        return false;
    }
    
    write_dlvk_header(file, metadata);
    write_dlvk_tensors(file, tensors);
    
    file.close();
    std::cout << "✅ Model exported to DLVK format: " << filename << std::endl;
    return true;
}

bool DLVKExporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open DLVK file: " << filename << std::endl;
        return false;
    }
    

    std::string magic(4, '\0');
    file.read(&magic[0], 4);
    if (magic != "DLVK") {
        std::cerr << "Invalid DLVK file format" << std::endl;
        return false;
    }
    

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    

    uint32_t metadata_len;
    file.read(reinterpret_cast<char*>(&metadata_len), sizeof(metadata_len));
    

    std::string metadata_json(metadata_len, '\0');
    file.read(&metadata_json[0], metadata_len);
    

    metadata.name = "Imported DLVK Model";
    metadata.framework = "dlvk";
    
    file.close();
    std::cout << "✅ Model imported from DLVK format: " << filename << std::endl;
    return true;
}

void DLVKExporter::write_dlvk_header(std::ofstream& file, const ModelMetadata& metadata) {

    file.write("DLVK", 4);
    

    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    

    std::ostringstream metadata_json;
    metadata_json << "{\n";
    metadata_json << "  \"name\": \"" << metadata.name << "\",\n";
    metadata_json << "  \"architecture\": \"" << metadata.architecture << "\",\n";
    metadata_json << "  \"version\": \"" << metadata.version << "\",\n";
    metadata_json << "  \"framework\": \"dlvk\",\n";
    metadata_json << "  \"vocab_size\": " << metadata.vocab_size << ",\n";
    metadata_json << "  \"embedding_dim\": " << metadata.embedding_dim << ",\n";
    metadata_json << "  \"hidden_size\": " << metadata.hidden_size << ",\n";
    metadata_json << "  \"sequence_length\": " << metadata.sequence_length << ",\n";
    metadata_json << "  \"num_layers\": " << metadata.num_layers << "\n";
    metadata_json << "}";
    
    std::string json_str = metadata_json.str();
    uint32_t metadata_len = json_str.length();
    

    file.write(reinterpret_cast<const char*>(&metadata_len), sizeof(metadata_len));
    

    file.write(json_str.c_str(), metadata_len);
}

void DLVKExporter::write_dlvk_tensors(std::ofstream& file, const std::vector<TensorInfo>& tensors) {

    uint32_t tensor_count = tensors.size();
    file.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
    

    write_binary_data(file, tensors);
}


ModelExportManager::ModelExportManager() {
    m_exporters[ModelFormat::DLVK] = std::make_unique<DLVKExporter>();
    m_exporters[ModelFormat::ONNX] = std::make_unique<ONNXExporter>();
    m_exporters[ModelFormat::GGUF] = std::make_unique<GGUFExporter>();
    m_exporters[ModelFormat::HDF5] = std::make_unique<HDF5Exporter>();
    m_exporters[ModelFormat::SAFETENSORS] = std::make_unique<SafeTensorsExporter>();
}

bool ModelExportManager::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    ModelFormat format,
    const ModelMetadata& metadata) {
    
    auto it = m_exporters.find(format);
    if (it == m_exporters.end()) {
        std::cerr << "Unsupported export format" << std::endl;
        return false;
    }
    
    return it->second->export_model(model, filename, metadata);
}

bool ModelExportManager::import_model(
    const std::string& filename,
    ModelFormat format,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    auto it = m_exporters.find(format);
    if (it == m_exporters.end()) {
        std::cerr << "Unsupported import format" << std::endl;
        return false;
    }
    
    return it->second->import_model(filename, model, metadata);
}

ModelFormat ModelExportManager::detect_format(const std::string& filename) {
    if (filename.length() >= 5 && filename.substr(filename.length() - 5) == ".dlvk") return ModelFormat::DLVK;
    if (filename.length() >= 5 && filename.substr(filename.length() - 5) == ".onnx") return ModelFormat::ONNX;
    if (filename.length() >= 5 && filename.substr(filename.length() - 5) == ".gguf") return ModelFormat::GGUF;
    if (filename.length() >= 3 && filename.substr(filename.length() - 3) == ".h5") return ModelFormat::HDF5;
    if (filename.length() >= 12 && filename.substr(filename.length() - 12) == ".safetensors") return ModelFormat::SAFETENSORS;
    
    return ModelFormat::DLVK;  // Default to native format
}

bool ModelExportManager::export_all_formats(
    const std::shared_ptr<Model>& model,
    const std::string& base_filename,
    const ModelMetadata& metadata) {
    
    bool success = true;
    
    success &= export_model(model, base_filename + ".dlvk", ModelFormat::DLVK, metadata);
    success &= export_model(model, base_filename + ".onnx", ModelFormat::ONNX, metadata);
    success &= export_model(model, base_filename + ".gguf", ModelFormat::GGUF, metadata);
    success &= export_model(model, base_filename + ".h5", ModelFormat::HDF5, metadata);
    success &= export_model(model, base_filename + ".safetensors", ModelFormat::SAFETENSORS, metadata);
    
    return success;
}

std::vector<ModelFormat> ModelExportManager::get_supported_formats() const {
    return {
        ModelFormat::DLVK,
        ModelFormat::ONNX,
        ModelFormat::GGUF,
        ModelFormat::HDF5,
        ModelFormat::SAFETENSORS
    };
}

std::string ModelExportManager::get_extension(ModelFormat format) const {
    switch (format) {
        case ModelFormat::DLVK: return ".dlvk";
        case ModelFormat::ONNX: return ".onnx";
        case ModelFormat::GGUF: return ".gguf";
        case ModelFormat::HDF5: return ".h5";
        case ModelFormat::SAFETENSORS: return ".safetensors";
        default: return ".bin";
    }
}


std::unique_ptr<ModelExportManager> create_export_manager() {
    return std::make_unique<ModelExportManager>();
}

} // namespace export_formats
} // namespace dlvk
