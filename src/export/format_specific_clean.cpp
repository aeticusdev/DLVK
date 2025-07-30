#include <dlvk/export/model_export.h>
#include <iostream>
#include <fstream>
#include <sstream>


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
    
    try {

        hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            std::cerr << "Failed to create HDF5 file: " << filename << std::endl;
            return false;
        }
        

        hid_t model_group = H5Gcreate2(file_id, "/model", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        

        write_hdf5_attribute(model_group, "name", metadata.name);
        write_hdf5_attribute(model_group, "architecture", metadata.architecture);
        write_hdf5_attribute(model_group, "version", metadata.version);
        write_hdf5_attribute(model_group, "framework", "dlvk");
        

        write_hdf5_attribute(model_group, "vocab_size", static_cast<int>(metadata.vocab_size));
        write_hdf5_attribute(model_group, "embedding_dim", static_cast<int>(metadata.embedding_dim));
        write_hdf5_attribute(model_group, "hidden_size", static_cast<int>(metadata.hidden_size));
        write_hdf5_attribute(model_group, "sequence_length", static_cast<int>(metadata.sequence_length));
        write_hdf5_attribute(model_group, "num_layers", static_cast<int>(metadata.num_layers));
        

        hid_t weights_group = H5Gcreate2(model_group, "weights", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        

        for (const auto& tensor : tensors) {
            write_hdf5_tensor(weights_group, tensor);
        }
        

        H5Gclose(weights_group);
        H5Gclose(model_group);
        H5Fclose(file_id);
        
        std::cout << "✅ Model exported to HDF5 format: " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to export HDF5 model: " << e.what() << std::endl;
        return false;
    }
}

bool HDF5Exporter::import_model(
    const std::string& filename,
    std::shared_ptr<Model>& model,
    ModelMetadata& metadata) {
    
    try {

        hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            std::cerr << "Failed to open HDF5 file: " << filename << std::endl;
            return false;
        }
        

        hid_t model_group = H5Gopen2(file_id, "/model", H5P_DEFAULT);
        

        metadata.name = read_hdf5_string_attribute(model_group, "name");
        metadata.architecture = read_hdf5_string_attribute(model_group, "architecture");
        metadata.version = read_hdf5_string_attribute(model_group, "version");
        metadata.framework = read_hdf5_string_attribute(model_group, "framework");
        

        metadata.vocab_size = read_hdf5_int_attribute(model_group, "vocab_size");
        metadata.embedding_dim = read_hdf5_int_attribute(model_group, "embedding_dim");
        metadata.hidden_size = read_hdf5_int_attribute(model_group, "hidden_size");
        metadata.sequence_length = read_hdf5_int_attribute(model_group, "sequence_length");
        metadata.num_layers = read_hdf5_int_attribute(model_group, "num_layers");
        

        

        H5Gclose(model_group);
        H5Fclose(file_id);
        
        std::cout << "✅ Model imported from HDF5 format: " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to import HDF5 model: " << e.what() << std::endl;
        return false;
    }
}

void HDF5Exporter::write_hdf5_attribute(hid_t group_id, const std::string& name, const std::string& value) {
    hid_t string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, value.length() + 1);
    H5Tset_strpad(string_type, H5T_STR_NULLTERM);
    
    hid_t space_id = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate2(group_id, name.c_str(), string_type, space_id, H5P_DEFAULT, H5P_DEFAULT);
    
    H5Awrite(attr_id, string_type, value.c_str());
    
    H5Aclose(attr_id);
    H5Sclose(space_id);
    H5Tclose(string_type);
}

void HDF5Exporter::write_hdf5_attribute(hid_t group_id, const std::string& name, int value) {
    hid_t space_id = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate2(group_id, name.c_str(), H5T_NATIVE_INT, space_id, H5P_DEFAULT, H5P_DEFAULT);
    
    H5Awrite(attr_id, H5T_NATIVE_INT, &value);
    
    H5Aclose(attr_id);
    H5Sclose(space_id);
}

void HDF5Exporter::write_hdf5_tensor(hid_t group_id, const TensorInfo& tensor) {
    if (!tensor.data) return;
    

    std::vector<hsize_t> dims;
    for (size_t dim : tensor.shape) {
        dims.push_back(static_cast<hsize_t>(dim));
    }
    
    hid_t space_id = H5Screate_simple(dims.size(), dims.data(), nullptr);
    hid_t dataset_id = H5Dcreate2(group_id, tensor.name.c_str(), H5T_NATIVE_FLOAT, space_id, 
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    

    std::vector<float> cpu_data;
    tensor.data->download_data(cpu_data);
    

    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, cpu_data.data());
    
    H5Dclose(dataset_id);
    H5Sclose(space_id);
}

std::string HDF5Exporter::read_hdf5_string_attribute(hid_t group_id, const std::string& name) {
    hid_t attr_id = H5Aopen(group_id, name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) return "";
    
    hid_t type_id = H5Aget_type(attr_id);
    size_t size = H5Tget_size(type_id);
    
    std::vector<char> buffer(size);
    H5Aread(attr_id, type_id, buffer.data());
    
    H5Tclose(type_id);
    H5Aclose(attr_id);
    
    return std::string(buffer.data());
}

int HDF5Exporter::read_hdf5_int_attribute(hid_t group_id, const std::string& name) {
    hid_t attr_id = H5Aopen(group_id, name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) return 0;
    
    int value;
    H5Aread(attr_id, H5T_NATIVE_INT, &value);
    
    H5Aclose(attr_id);
    return value;
}

#else

bool HDF5Exporter::export_model(
    const std::shared_ptr<Model>& model,
    const std::string& filename,
    const ModelMetadata& metadata) {
    
    std::cerr << "HDF5 export not supported - HDF5 library not available" << std::endl;
    return false;
}

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
