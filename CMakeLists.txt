cmake_minimum_required(VERSION 3.10)
project(PybindExample LANGUAGES CXX)
# Enable AVX, AVX2, and FMA instructions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma")

cmake_minimum_required(VERSION 3.18)
project(PybindMatrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CUDA build
option(BUILD_CUDA "Build GPU bindings using CUDA" OFF)

# pybind11
add_subdirectory(extern/pybind11)

# Always build CPU bindings
add_subdirectory(cpu)

# Optionally build GPU bindings
if(BUILD_CUDA)
    enable_language(CUDA)
    # Ensure that CUDA 10.1 or above is being used
    find_package(CUDAToolkit REQUIRED)

    # Require CUDA ≥ 10.1
    if (CUDAToolkit_VERSION VERSION_LESS "10.1")
        message(FATAL_ERROR
                "CUDA Toolkit version ${CUDAToolkit_VERSION} found, "
                "but version 10.1 or greater is required")
    endif()
    add_subdirectory(gpu)
endif()