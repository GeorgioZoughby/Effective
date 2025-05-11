#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream> // Include for std::stringstream
#include "../include/Matrix.h"
#include "../include/MatrixTranspose.h" // Include MatrixTranspose header

namespace py = pybind11;

template <typename T>
void bind_matrix(py::module &m, const std::string &type_name)
{
    py::class_<Matrix<T>>(m, type_name.c_str())
        .def(py::init<std::size_t, std::size_t>())
        .def("rows", &Matrix<T>::rows)
        .def("columns", &Matrix<T>::columns)
        .def("__call__", static_cast<T &(Matrix<T>::*)(std::size_t, std::size_t)>(&Matrix<T>::operator())) // Resolve overload
        .def("setZero", &Matrix<T>::setZero)
        .def("setOnes", &Matrix<T>::setOnes)
        .def("setConstant", &Matrix<T>::setConstant)
        .def("sum", &Matrix<T>::sum)
        .def("transpose", &Matrix<T>::transpose)
        .def("dot", &Matrix<T>::dot)
        .def("normalized", &Matrix<T>::normalized)
        .def("__repr__", [](const Matrix<T> &m)
             {
            std::stringstream ss;
            ss << m;
            return ss.str(); });

    // Add bindings for MatrixTranspose with proper constructor and methods
    py::class_<MatrixTranspose<T>>(m, ("MatrixTranspose" + type_name).c_str())
        .def(py::init<const Matrix<T> &>()) // Constructor
        .def("rows", &MatrixTranspose<T>::rows)
        .def("columns", &MatrixTranspose<T>::columns)
        .def("__call__", [](const MatrixTranspose<T> &self, std::size_t row, std::size_t col)
             { return self(row, col); })
        .def("__repr__", [](const MatrixTranspose<T> &transpose)
             {
            std::stringstream ss;
            ss << "MatrixTranspose with " << transpose.rows() << " rows and " << transpose.columns() << " columns.";
            return ss.str(); });
}

PYBIND11_MODULE(matrix_module, m)
{
    bind_matrix<int>(m, "MatrixInt");
    bind_matrix<float>(m, "MatrixFloat");
    bind_matrix<double>(m, "MatrixDouble");
}
