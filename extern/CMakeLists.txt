add_library(cuda_matrix_impl STATIC
        ../include/MatrixCuda.cu
)
target_include_directories(cuda_matrix_impl PUBLIC
        ../include
)
pybind11_add_module(matrix_gpu
        ../bindings/matrix_bindings_gpu.cu
        ../include/MatrixCuda.cu
)
set_target_properties(matrix_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        PREFIX ""
        SUFFIX ".pyd"
)

target_link_libraries(matrix_gpu PRIVATE
        pybind11::module
        CUDA::cudart
)

target_include_directories(matrix_gpu PRIVATE
        ../include
)

