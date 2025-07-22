file(REMOVE_RECURSE
  "CMakeFiles/shaders"
  "shaders/matrix_multiply.comp.spv"
  "shaders/reduce_sum.comp.spv"
  "shaders/relu.comp.spv"
  "shaders/sigmoid.comp.spv"
  "shaders/softmax.comp.spv"
  "shaders/tanh.comp.spv"
  "shaders/tensor_add.comp.spv"
  "shaders/tensor_divide.comp.spv"
  "shaders/tensor_multiply.comp.spv"
  "shaders/tensor_subtract.comp.spv"
  "shaders/transpose.comp.spv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/shaders.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
