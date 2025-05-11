cmake_minimum_required(VERSION 3.10)
project(PybindExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AVX, AVX2, and FMA instructions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma")

add_subdirectory(extern/pybind11)
pybind11_add_module(matrix_cpu bindings/matrix_bindings_cpu.cpp)
pybind11_add_module(vector_cpu bindings/vector_bindings_cpu.cpp)
