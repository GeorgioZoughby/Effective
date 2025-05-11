cmake_minimum_required(VERSION 3.10)

project(PybindExample)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Include Python headers and libraries
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARIES})

set(pybind11_DIR /home/georgio/Desktop/Cpp/Project_Effective/env/lib/python3.13/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)

add_library(python_module SHARED bindings/matrix_bindings_cpu.cpp)

# Link Python libraries
set(PYTHON_LIBRARIES ${Python3_LIBRARIES})

# Link pybind11 and Python libraries to your program
target_link_libraries(python_module PRIVATE pybind11::module ${PYTHON_LIBRARIES})
