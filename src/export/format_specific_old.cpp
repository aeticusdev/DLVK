#include "dlvk/export/model_export.h"
#include <iostream>


#ifdef DLVK_HDF5_SUPPORT
#include <hdf5.h>
#endif






namespace dlvk {
namespace export_formats {


#ifdef DLVK_HDF5_SUPPORT
bool HDF5Exporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    auto tensors = extract_tensors(model);
    

    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Failed to create HDF5 file: " << filename << std::endl;
        return false;
    }
    

    hid_t metadata_group = H5Gcreate2(file_id, "/metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    write_hdf5_attribute(metadata_group, "name", metadata.name);
    write_hdf5_attribute(metadata_group, "architecture", metadata.architecture);
    write_hdf5_attribute(metadata_group, "version", metadata.version);
    write_hdf5_attribute(metadata_group, "vocab_size", static_cast<int>(metadata.vocab_size));
    write_hdf5_attribute(metadata_group, "embedding_dim", static_cast<int>(metadata.embedding_dim));
    write_hdf5_attribute(metadata_group, "hidden_size", static_cast<int>(metadata.hidden_size));
    write_hdf5_attribute(metadata_group, "sequence_length", static_cast<int>(metadata.sequence_length));
    
    H5Gclose(metadata_group);
    

    hid_t tensors_group = H5Gcreate2(file_id, "/tensors", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    for (const auto& tensor : tensors) {
        write_hdf5_tensor(tensors_group, tensor);
    }
    
    H5Gclose(tensors_group);
    H5Fclose(file_id);
    
    std::cout << "✅ Model exported to HDF5 format: " << filename << std::endl;
    return true;
}
#else
bool HDF5Exporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    std::cerr << "HDF5 export not supported - HDF5 library not available" << std::endl;
    return false;
}
#endif

#ifdef DLVK_HDF5_SUPPORT
bool HDF5Exporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    

    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Failed to open HDF5 file: " << filename << std::endl;
        return false;
    }
    

    hid_t metadata_group = H5Gopen2(file_id, "/metadata", H5P_DEFAULT);
    
    metadata.name = read_hdf5_string_attribute(metadata_group, "name");
    metadata.architecture = read_hdf5_string_attribute(metadata_group, "architecture");
    metadata.version = read_hdf5_string_attribute(metadata_group, "version");
    metadata.vocab_size = read_hdf5_int_attribute(metadata_group, "vocab_size");
    metadata.embedding_dim = read_hdf5_int_attribute(metadata_group, "embedding_dim");
    metadata.hidden_size = read_hdf5_int_attribute(metadata_group, "hidden_size");
    metadata.sequence_length = read_hdf5_int_attribute(metadata_group, "sequence_length");
    
    H5Gclose(metadata_group);
    

    hid_t tensors_group = H5Gopen2(file_id, "/tensors", H5P_DEFAULT);
    

    
    H5Gclose(tensors_group);
    H5Fclose(file_id);
    
    std::cout << "✅ Model imported from HDF5 format: " << filename << std::endl;
    return true;
}

void HDF5Exporter::write_hdf5_attribute(hid_t group_id, const std::string& name, const std::string& value) {
    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t datatype = H5Tcopy(H5T_C_S1);
    H5Tset_size(datatype, value.length() + 1);
    H5Tset_strpad(datatype, H5T_STR_NULLTERM);
    
    hid_t attribute = H5Acreate2(group_id, name.c_str(), datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, datatype, value.c_str());
    
    H5Aclose(attribute);
    H5Tclose(datatype);
    H5Sclose(dataspace);
}

void HDF5Exporter::write_hdf5_attribute(hid_t group_id, const std::string& name, int value) {
    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t attribute = H5Acreate2(group_id, name.c_str(), H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, H5T_NATIVE_INT, &value);
    
    H5Aclose(attribute);
    H5Sclose(dataspace);
}

void HDF5Exporter::write_hdf5_tensor(hid_t group_id, const TensorInfo& tensor) {
    std::vector<hsize_t> dims(tensor.shape.begin(), tensor.shape.end());
    hid_t dataspace = H5Screate_simple(dims.size(), dims.data(), nullptr);
    
    hid_t dataset = H5Dcreate2(group_id, tensor.name.c_str(), H5T_NATIVE_FLOAT, dataspace,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    if (tensor.data) {

        std::vector<float> buffer(tensor.size_bytes / sizeof(float));
        tensor.data->download_data(buffer.data());
        H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());
    }
    
    H5Dclose(dataset);
    H5Sclose(dataspace);
}

std::string HDF5Exporter::read_hdf5_string_attribute(hid_t group_id, const std::string& name) {
    hid_t attribute = H5Aopen(group_id, name.c_str(), H5P_DEFAULT);
    hid_t datatype = H5Aget_type(attribute);
    size_t size = H5Tget_size(datatype);
    
    std::vector<char> buffer(size);
    H5Aread(attribute, datatype, buffer.data());
    
    H5Tclose(datatype);
    H5Aclose(attribute);
    
    return std::string(buffer.data());
}

int HDF5Exporter::read_hdf5_int_attribute(hid_t group_id, const std::string& name) {
    hid_t attribute = H5Aopen(group_id, name.c_str(), H5P_DEFAULT);
    int value;
    H5Aread(attribute, H5T_NATIVE_INT, &value);
    H5Aclose(attribute);
    return value;
}
#else
bool HDF5Exporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    std::cerr << "HDF5 import not supported - HDF5 library not available" << std::endl;
    return false;
}

void HDF5Exporter::write_hdf5_attribute(hid_t, const std::string&, const std::string&) {}
void HDF5Exporter::write_hdf5_attribute(hid_t, const std::string&, int) {}
void HDF5Exporter::write_hdf5_tensor(hid_t, const TensorInfo&) {}
std::string HDF5Exporter::read_hdf5_string_attribute(hid_t, const std::string&) { return ""; }
int HDF5Exporter::read_hdf5_int_attribute(hid_t, const std::string&) { return 0; }
#endif

} // namespace export_formats
} // namespace dlvk


#ifdef DLVK_PYTORCH_SUPPORT
bool PyTorchExporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    auto tensors = extract_tensors(model);
    
    try {
        torch::serialize::OutputArchive archive;
        

        std::string name = metadata.name;
        std::string architecture = metadata.architecture;
        std::string version = metadata.version;
        archive.write("metadata/name", name);
        archive.write("metadata/architecture", architecture);
        archive.write("metadata/version", version);
        archive.write("metadata/vocab_size", static_cast<int64_t>(metadata.vocab_size));
        archive.write("metadata/embedding_dim", static_cast<int64_t>(metadata.embedding_dim));
        archive.write("metadata/hidden_size", static_cast<int64_t>(metadata.hidden_size));
        archive.write("metadata/sequence_length", static_cast<int64_t>(metadata.sequence_length));
        

        archive.write("tensor_count", static_cast<int64_t>(tensors.size()));
        

        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto& tensor = tensors[i];
            

            std::vector<int64_t> torch_shape(tensor.shape.begin(), tensor.shape.end());
            

            torch::Tensor torch_tensor;
            if (tensor.data) {

                std::vector<float> buffer(tensor.size_bytes / sizeof(float));
                tensor.data->download_data(buffer.data());
                
                torch_tensor = torch::from_blob(
                    buffer.data(),
                    torch_shape,
                    torch::kFloat32
                ).clone();
            } else {
                torch_tensor = torch::zeros(torch_shape, torch::kFloat32);
            }
            

            std::string tensor_key = "tensors/" + tensor.name;
            archive.write(tensor_key, torch_tensor);
        }
        

        archive.save_to(filename);
        
        std::cout << "✅ Model exported to PyTorch format: " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to export PyTorch model: " << e.what() << std::endl;
        return false;
    }
}

bool PyTorchExporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(filename);
        

        std::string name, architecture, version;
        archive.read("metadata/name", name);
        archive.read("metadata/architecture", architecture);
        archive.read("metadata/version", version);
        
        metadata.name = name;
        metadata.architecture = architecture;
        metadata.version = version;
        
        int64_t vocab_size, embedding_dim, hidden_size, sequence_length;
        archive.read("metadata/vocab_size", vocab_size);
        archive.read("metadata/embedding_dim", embedding_dim);
        archive.read("metadata/hidden_size", hidden_size);
        archive.read("metadata/sequence_length", sequence_length);
        
        metadata.vocab_size = static_cast<size_t>(vocab_size);
        metadata.embedding_dim = static_cast<size_t>(embedding_dim);
        metadata.hidden_size = static_cast<size_t>(hidden_size);
        metadata.sequence_length = static_cast<size_t>(sequence_length);
        

        int64_t tensor_count;
        archive.read("tensor_count", tensor_count);
        

        
        std::cout << "✅ Model imported from PyTorch format: " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to import PyTorch model: " << e.what() << std::endl;
        return false;
    }
}
#else
bool PyTorchExporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    std::cerr << "PyTorch export not supported - PyTorch library not available" << std::endl;
    return false;
}

bool PyTorchExporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    std::cerr << "PyTorch import not supported - PyTorch library not available" << std::endl;
    return false;
}
#endif

} // namespace export_formats
} // namespace dlvk
