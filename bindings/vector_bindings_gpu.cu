#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream> // Include for std::stringstream
#include "../include/VectorCuda.cuh"
#include "../include/VectorCuda.cu"

namespace py = pybind11;

template <typename T>
void bind_vector(py::module &m, const std::string &type_name)
{
    using VectorType = Vector<T>;
    using namespace expr;
    using VectorAddType = VectorAdd<T>;
    using VectorSubType = VectorSub<T>;
    using VectorMulType = VectorMul<T>;

    // Vector class
    py::class_<VectorType>(m, type_name.c_str())
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<const VectorType &>())
        .def("resize", &VectorType::resize)
        .def("size", &VectorType::size)
        .def("capacity", &VectorType::capacity)
        .def("empty", &VectorType::empty)
        .def("reserve", &VectorType::reserve)
        .def("clear", &VectorType::clear)
        .def("push_back", &VectorType::push_back)
        .def("pop_back", &VectorType::pop_back)
        .def("at", static_cast<T &(VectorType::*)(int)>(&VectorType::at))
        .def("at", static_cast<const T &(VectorType::*)(int) const>(&VectorType::at))
        .def("__getitem__", [](const VectorType &self, size_t i)
             {
            if (i >= self.size()) throw py::index_error();
            return self[i]; })
        .def("__setitem__", [](VectorType &self, size_t i, T val)
             {
            if (i >= self.size()) throw py::index_error();
            self[i] = val; })
        .def("front", static_cast<T &(VectorType::*)()>(&VectorType::front))
        .def("front", static_cast<const T &(VectorType::*)() const>(&VectorType::front))
        .def("back", static_cast<T &(VectorType::*)()>(&VectorType::back))
        .def("back", static_cast<const T &(VectorType::*)() const>(&VectorType::back))
        .def("data", static_cast<T *(VectorType::*)()>(&VectorType::data))
        .def("data", static_cast<const T *(VectorType::*)() const>(&VectorType::data))
        .def("dot", [](const VectorType &self, const VectorType &other)
             {
            // Implement a safer version of dot product to avoid segmentation fault
            if (self.size() != other.size()) {
                throw std::invalid_argument("Vectors must be the same size for dot product");
            }
            T result = T();
            for (size_t i = 0; i < self.size(); ++i) {
                result += self[i] * other[i];
            }
            return result; })
        // Scalar operations
        .def("__add__", [](const VectorType &self, T scalar)
             {
            VectorType result(self);
            result += scalar;
            return result; })
        .def("__radd__", [](const VectorType &self, T scalar)
             {
            VectorType result(self);
            result += scalar;
            return result; })
        .def("__sub__", [](const VectorType &self, T scalar)
             {
            VectorType result(self);
            result -= scalar;
            return result; })
        .def("__rsub__", [](const VectorType &self, T scalar)
             {
            VectorType result(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                result[i] = scalar - self[i];
            }
            return result; })
        .def("__mul__", [](const VectorType &self, T scalar)
             {
            VectorType result(self);
            result *= scalar;
            return result; })
        .def("__rmul__", [](const VectorType &self, T scalar)
             {
            VectorType result(self);
            result *= scalar;
            return result; })
        .def("__truediv__", [](const VectorType &self, T scalar)
             {
            VectorType result(self);
            result /= scalar;
            return result; })
        // Vector operations
        .def("__add__", [](const VectorType &self, const VectorType &other)
             { return VectorAddType(self, other); })
        .def("__sub__", [](const VectorType &self, const VectorType &other)
             { return VectorSubType(self, other); })
        .def("__mul__", [](const VectorType &self, const VectorType &other)
             { return VectorMulType(self, other); })
        .def("__len__", &VectorType::size)
        .def("__repr__", [type_name](const VectorType &self)
             {
            std::stringstream ss;
            ss << type_name << "(size=" << self.size() << ", [";
            for (size_t i = 0; i < self.size(); ++i) {
                if (i > 0) ss << ", ";
                if (i >= 10) {
                    ss << "...";
                    break;
                }
                ss << self[i];
            }
            ss << "])";
            return ss.str(); })
        .def("__str__", [](const VectorType &self)
             {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < self.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << self[i];
            }
            ss << "]";
            return ss.str(); });
}

// Register the VectorAdd expression type
template <typename T>
void register_vector_add(py::module &m, const std::string &type_name)
{
    using namespace expr;
    using VectorAddType = VectorAdd<T>;
    using VectorSubType = VectorSub<T>;
    using VectorMulType = VectorMul<T>;
    using VectorType = Vector<T>;

    // VectorAdd
    py::class_<VectorAddType>(m, ("VectorAdd" + type_name).c_str())
        .def("size", &VectorAddType::size)
        .def("__getitem__", [](const VectorAddType &self, size_t i)
             {
            if (i >= self.size()) throw py::index_error();
            return self[i]; })
        .def("eval", [](const VectorAddType &expr)
             {
            VectorType result(expr.size());
            result.resize(expr.size());
            for (size_t i = 0; i < expr.size(); ++i) {
                result[i] = expr[i];
            }
            return result; })
        // Add operations for expression chaining
        .def("__add__", [](const VectorAddType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorAddType(self_eval, other); })
        .def("__sub__", [](const VectorAddType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorSubType(self_eval, other); })
        .def("__mul__", [](const VectorAddType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorMulType(self_eval, other); })
        .def("__repr__", [](const VectorAddType &expr)
             {
            std::stringstream ss;
            ss << "Vector Expression (size=" << expr.size() << ")";
            return ss.str(); })
        .def("__str__", [](const VectorAddType &expr)
             {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < expr.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << expr[i];
            }
            ss << "]";
            return ss.str(); });
}

// Register the VectorSub expression type
template <typename T>
void register_vector_sub(py::module &m, const std::string &type_name)
{
    using namespace expr;
    using VectorAddType = VectorAdd<T>;
    using VectorSubType = VectorSub<T>;
    using VectorMulType = VectorMul<T>;
    using VectorType = Vector<T>;

    // VectorSub
    py::class_<VectorSubType>(m, ("VectorSub" + type_name).c_str())
        .def("size", &VectorSubType::size)
        .def("__getitem__", [](const VectorSubType &self, size_t i)
             {
            if (i >= self.size()) throw py::index_error();
            return self[i]; })
        .def("eval", [](const VectorSubType &expr)
             {
            VectorType result(expr.size());
            result.resize(expr.size());
            for (size_t i = 0; i < expr.size(); ++i) {
                result[i] = expr[i];
            }
            return result; })
        // Add operations for expression chaining
        .def("__add__", [](const VectorSubType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorAddType(self_eval, other); })
        .def("__sub__", [](const VectorSubType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorSubType(self_eval, other); })
        .def("__mul__", [](const VectorSubType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorMulType(self_eval, other); })
        .def("__repr__", [](const VectorSubType &expr)
             {
            std::stringstream ss;
            ss << "Vector Expression (size=" << expr.size() << ")";
            return ss.str(); })
        .def("__str__", [](const VectorSubType &expr)
             {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < expr.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << expr[i];
            }
            ss << "]";
            return ss.str(); });
}

// Register the VectorMul expression type
template <typename T>
void register_vector_mul(py::module &m, const std::string &type_name)
{
    using namespace expr;
    using VectorAddType = VectorAdd<T>;
    using VectorSubType = VectorSub<T>;
    using VectorMulType = VectorMul<T>;
    using VectorType = Vector<T>;

    // VectorMul
    py::class_<VectorMulType>(m, ("VectorMul" + type_name).c_str())
        .def("size", &VectorMulType::size)
        .def("__getitem__", [](const VectorMulType &self, size_t i)
             {
            if (i >= self.size()) throw py::index_error();
            return self[i]; })
        .def("eval", [](const VectorMulType &expr)
             {
            VectorType result(expr.size());
            result.resize(expr.size());
            for (size_t i = 0; i < expr.size(); ++i) {
                result[i] = expr[i];
            }
            return result; })
        // Add operations for expression chaining
        .def("__add__", [](const VectorMulType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorAddType(self_eval, other); })
        .def("__sub__", [](const VectorMulType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorSubType(self_eval, other); })
        .def("__mul__", [](const VectorMulType &self, const VectorType &other)
             {
            VectorType self_eval(self.size());
            self_eval.resize(self.size());
            for (size_t i = 0; i < self.size(); ++i) {
                self_eval[i] = self[i];
            }
            return VectorMulType(self_eval, other); })
        .def("__repr__", [](const VectorMulType &expr)
             {
            std::stringstream ss;
            ss << "Vector Expression (size=" << expr.size() << ")";
            return ss.str(); })
        .def("__str__", [](const VectorMulType &expr)
             {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < expr.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << expr[i];
            }
            ss << "]";
            return ss.str(); });
}

PYBIND11_MODULE(vector_gpu, m)
{
    bind_vector<int>(m, "VectorInt");
    bind_vector<float>(m, "VectorFloat");
    // bind_vector<double>(m, "VectorDouble");

    // Register the expression types
    register_vector_add<int>(m, "Int");
    register_vector_add<float>(m, "Float");
    // register_vector_add<double>(m, "Double");

    register_vector_sub<int>(m, "Int");
    register_vector_sub<float>(m, "Float");
    // register_vector_sub<double>(m, "Double");

    register_vector_mul<int>(m, "Int");
    register_vector_mul<float>(m, "Float");
    // register_vector_mul<double>(m, "Double");
}