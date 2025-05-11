cmake_minimum_required(VERSION 3.10)
project(PybindExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(extern/pybind11)
pybind11_add_module(matrix_cpu bindings/matrix_bindings_cpu.cpp)
