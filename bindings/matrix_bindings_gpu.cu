#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream> // Include for std::stringstream
#include "../include/MatrixCuda.cuh"
#include "../include/MatrixCuda.cu"

namespace py = pybind11;

template <typename T>
void bind_matrix(py::module &m, const std::string &type_name)
{
    using MatrixType = Matrix<T>;
    using MatrixTransposeType = MatrixTranspose<T>;
    using MatrixAddType = MatrixAdd<T>;
    using MatrixMulType = MatrixMul<T>;
    using MatrixSubType = MatrixSub<T>;

    // Matrix class
    py::class_<MatrixType>(m, type_name.c_str())
        .def(py::init<>())
        .def(py::init<std::size_t, std::size_t>())
        .def(py::init<const MatrixType &>())
        .def("resize", &MatrixType::resize)
        .def("rows", &MatrixType::rows)
        .def("columns", &MatrixType::columns)
        .def("__call__", static_cast<T &(MatrixType::*)(std::size_t, std::size_t)>(&MatrixType::operator()))
        .def("__call__", [](const MatrixType &self, std::size_t row, std::size_t col)
             { return self(row, col); })
        // Keep the set method for Python compatibility
        .def("set", [](MatrixType &self, std::size_t row, std::size_t col, T value)
             { self(row, col) = value; })
        // Add methods from old bindings
        .def("setZero", &MatrixType::setZero)
        .def("setOnes", &MatrixType::setOnes)
        .def("setConstant", &MatrixType::setConstant)
        .def("sum", &MatrixType::sum)
        .def("transpose", &MatrixType::transpose)
        // Fix the dot method to explicitly accept Matrix<T> instead of MatrixExpression<T>
        .def("dot", [](const MatrixType &self, const MatrixType &other)
             { return self.dot(other); })
        .def("normalized", &MatrixType::normalized)
        // Add the operators
        .def("__add__", [](const MatrixType &self, const MatrixType &other)
             { return MatrixAddType(self, other); })
        .def("__sub__", [](const MatrixType &self, const MatrixType &other)
             { return MatrixSubType(self, other); })
        .def("__mul__", [](const MatrixType &self, const MatrixType &other)
             { return MatrixMulType(self, other); })
        .def("__str__", [](const MatrixType &self)
             {
            std::stringstream ss;
            ss << self;
            return ss.str(); })
        .def("__repr__", [](const MatrixType &self)
             {
            std::stringstream ss;
            ss << self;
            return ss.str(); });

    // MatrixTranspose - using the original class name format
    py::class_<MatrixTransposeType>(m, ("MatrixTranspose" + type_name).c_str())
        .def(py::init<const MatrixType &>()) // Constructor
        .def("rows", &MatrixTransposeType::rows)
        .def("columns", &MatrixTransposeType::columns)
        .def("__call__", [](const MatrixTransposeType &self, std::size_t row, std::size_t col)
             { return self(row, col); })
        .def("__repr__", [](const MatrixTransposeType &transpose)
             {
            std::stringstream ss;
            ss << "MatrixTranspose with " << transpose.rows() << " rows and " << transpose.columns() << " columns.";
            return ss.str(); })
        .def("eval", [](const MatrixTransposeType &expr)
             {
            MatrixType result(expr.rows(), expr.columns());
            for (std::size_t i = 0; i < expr.rows(); ++i) {
                for (std::size_t j = 0; j < expr.columns(); ++j) {
                    result(i, j) = expr(i, j);
                }
            }
            return result; })
        .def("__str__", [](const MatrixTransposeType &expr)
             {
            MatrixType result(expr.rows(), expr.columns());
            for (std::size_t i = 0; i < expr.rows(); ++i) {
                for (std::size_t j = 0; j < expr.columns(); ++j) {
                    result(i, j) = expr(i, j);
                }
            }
            std::stringstream ss;
            ss << result;
            return ss.str(); });
}

// Only registering MatrixAdd and not MatrixTranspose (already registered in bind_matrix)
template <typename T>
void register_matrix_add(py::module &m, const std::string &type_name)
{
    using MatrixAddType = MatrixAdd<T>;
    using MatrixType = Matrix<T>;

    // MatrixAdd
    py::class_<MatrixAddType>(m, ("MatrixAdd" + type_name).c_str())
        .def("rows", &MatrixAddType::rows)
        .def("columns", &MatrixAddType::columns)
        .def("__call__", [](const MatrixAddType &self, std::size_t row, std::size_t col)
             { return self(row, col); })
        .def("eval", [](const MatrixAddType &expr)
             {
            MatrixType result(expr.rows(), expr.columns());
            for (std::size_t i = 0; i < expr.rows(); ++i) {
                for (std::size_t j = 0; j < expr.columns(); ++j) {
                    result(i, j) = expr(i, j);
                }
            }
            return result; })
        .def("__repr__", [](const MatrixAddType &expr)
             {
            std::stringstream ss;
            ss << "Matrix Expression (" << expr.rows() << "x" << expr.columns() << ")";
            return ss.str(); })
        .def("__str__", [](const MatrixAddType &expr)
             {
            MatrixType result(expr.rows(), expr.columns());
            for (std::size_t i = 0; i < expr.rows(); ++i) {
                for (std::size_t j = 0; j < expr.columns(); ++j) {
                    result(i, j) = expr(i, j);
                }
            }
            std::stringstream ss;
            ss << result;
            return ss.str(); });
}

// Register the MatrixMul expression type
template <typename T>
void register_matrix_mul(py::module &m, const std::string &type_name)
{
    using MatrixMulType = MatrixMul<T>;
    using MatrixType = Matrix<T>;

    // MatrixMul
    py::class_<MatrixMulType>(m, ("MatrixMul" + type_name).c_str())
        .def("rows", &MatrixMulType::rows)
        .def("columns", &MatrixMulType::columns)
        .def("__call__", [](const MatrixMulType &self, std::size_t row, std::size_t col)
             { return self(row, col); })
        .def("eval", [](const MatrixMulType &expr)
             {
            MatrixType result(expr.rows(), expr.columns());
            for (std::size_t i = 0; i < expr.rows(); ++i) {
                for (std::size_t j = 0; j < expr.columns(); ++j) {
                    result(i, j) = expr(i, j);
                }
            }
            return result; })
        .def("__repr__", [](const MatrixMulType &expr)
             {
            std::stringstream ss;
            ss << "Matrix Expression (" << expr.rows() << "x" << expr.columns() << ")";
            return ss.str(); })
        .def("__str__", [](const MatrixMulType &expr)
             {
            MatrixType result(expr.rows(), expr.columns());
            for (std::size_t i = 0; i < expr.rows(); ++i) {
                for (std::size_t j = 0; j < expr.columns(); ++j) {
                    result(i, j) = expr(i, j);
                }
            }
            std::stringstream ss;
            ss << result;
            return ss.str(); });
}

// Register the MatrixSub expression type
template <typename T>
void register_matrix_sub(py::module &m, const std::string &type_name)
{
    using MatrixSubType = MatrixSub<T>;
    using MatrixType = Matrix<T>;

    // MatrixSub
    py::class_<MatrixSubType>(m, ("MatrixSub" + type_name).c_str())
        .def("rows", &MatrixSubType::rows)
        .def("columns", &MatrixSubType::columns)
        .def("__call__", [](const MatrixSubType &self, std::size_t row, std::size_t col)
             { return self(row, col); })
        .def("eval", [](const MatrixSubType &expr)
             {
            MatrixType result(expr.rows(), expr.columns());
            for (std::size_t i = 0; i < expr.rows(); ++i) {
                for (std::size_t j = 0; j < expr.columns(); ++j) {
                    result(i, j) = expr(i, j);
                }
            }
            return result; })
        .def("__repr__", [](const MatrixSubType &expr)
             {
            std::stringstream ss;
            ss << "Matrix Expression (" << expr.rows() << "x" << expr.columns() << ")";
            return ss.str(); })
        .def("__str__", [](const MatrixSubType &expr)
             {
            MatrixType result(expr.rows(), expr.columns());
            for (std::size_t i = 0; i < expr.rows(); ++i) {
                for (std::size_t j = 0; j < expr.columns(); ++j) {
                    result(i, j) = expr(i, j);
                }
            }
            std::stringstream ss;
            ss << result;
            return ss.str(); });
}

PYBIND11_MODULE(matrix_gpu, m)
{
    bind_matrix<int>(m, "MatrixInt");
    bind_matrix<float>(m, "MatrixFloat");
//    bind_matrix<double>(m, "MatrixDouble");

    // Register the expression types
    register_matrix_add<int>(m, "Int");
    register_matrix_add<float>(m, "Float");
//    register_matrix_add<double>(m, "Double");

    register_matrix_sub<int>(m, "Int");
    register_matrix_sub<float>(m, "Float");
//    register_matrix_sub<double>(m, "Double");

    register_matrix_mul<int>(m, "Int");
    register_matrix_mul<float>(m, "Float");
//    register_matrix_mul<double>(m, "Double");
}