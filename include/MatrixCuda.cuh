#ifndef MATRIXCUDA_CUH
#define MATRIXCUDA_CUH

// Helper macro to define stream and clean up
#define CREATE_CUDA_STREAM(stream) \
    cudaStream_t stream; \
    cudaStreamCreate(&stream);

#define DESTROY_CUDA_STREAM(stream) \
    cudaStreamSynchronize(stream); \
    cudaStreamDestroy(stream);


#include <iostream>
#include <algorithm>
#include <random>
#include <cassert>
#include <numeric>
#include <cmath>
#include <functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Forward declarations
template <typename T>
class MatrixExpression;
template <typename T>
class MatrixTranspose;
template <typename T>
class MatrixCwiseProduct;
template <typename T>
class MatrixCwiseQuotient;
template <typename T>
class MatrixRowView;
template <typename T>
class MatrixColView;
template <typename T>
class MatrixBlockView;

// Matrix Expression Interface
template <typename T>
class MatrixExpression {
public:
    virtual T operator()(std::size_t row, std::size_t col) const = 0;
    virtual std::size_t rows() const = 0;
    virtual std::size_t columns() const = 0;
    virtual ~MatrixExpression() = default;
};

// CUDA Operator Templates
template <typename T>
class AddOp {
public:
    __device__ void operator()(T& a, T b) const {
        a += b;
    }
};

template <typename T>
class SubtractOp {
public:
    __device__ void operator()(T& a, T b) const {
        a -= b;
    }
};

template <typename T>
class MultiplyOp {
public:
    __device__ void operator()(T& a, T b) const {
        a *= b;
    }
};

template <typename T>
class DivideOp {
public:
    __device__ void operator()(T& a, T b) const {
        a /= b;
    }
};

// CUDA Kernel Function Declarations
template <typename T>
__global__ void copyKernel(const T* from, T* to, int size);

template <typename Op, typename T>
__global__ void scalarOpKernel(T* elts, const T scal, int size, Op op);

template <typename T>
__global__ void compareEqKernel(const T* elts1, const T* elts2, bool* equality, int size);

template <typename T>
__global__ void dotKernel(const T* elts1, const T* elts2, T* result, unsigned int size);

template <typename T>
__global__ void clearKernel(T* elts, int size);

template <typename Op, typename T>
__global__ void matrixOpKernel(T* elts1, const T* elts2, unsigned int size, Op op);

template <typename T>
__global__ void matrixMultiplyKernel(const T* A, const T* B, T* C, int rowsA, int colsA, int colsB);

template <typename T>
__global__ void transposeKernel(const T* input, T* output, int rows, int cols);

// CUDA Interface Function Declarations
template <typename T>
cudaError_t cuda_matrix_copy(const T* from, T* to, unsigned int size);

template <typename Op, typename T>
cudaError_t cuda_matrix_scalar_op(T* elts, const T scalar, unsigned int size, Op op);

template <typename T>
cudaError_t cuda_matrix_compare_equality(const T* elts1, const T* elts2, unsigned int size, bool* equality);

template <typename T>
cudaError_t cuda_matrix_dot(const T* elts1, const T* elts2, unsigned int size, T* result);

template <typename T>
cudaError_t cuda_matrix_clear(T* elts, unsigned int size);

template <typename Op, typename T>
cudaError_t cuda_matrix_op(T* elts1, const T* elts2, unsigned int size, Op op);

template <typename T>
cudaError_t cuda_matrix_multiply(const T* A, const T* B, T* C, int rowsA, int colsA, int colsB);

template <typename T>
cudaError_t cuda_matrix_transpose(const T* input, T* output, int rows, int cols);

// Matrix Expression Classes
template <typename T>
class MatrixTranspose : public MatrixExpression<T> {
public:
    MatrixTranspose(const MatrixExpression<T>& mat) : _mat(mat) {}

    T operator()(std::size_t row, std::size_t col) const override {
        return _mat(col, row);
    }

    std::size_t rows() const override { return _mat.columns(); }
    std::size_t columns() const override { return _mat.rows(); }

private:
    const MatrixExpression<T>& _mat;
};

template <typename T>
class MatrixCwiseProduct : public MatrixExpression<T> {
public:
    MatrixCwiseProduct(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs)
        : _lhs(lhs), _rhs(rhs) {
        assert(lhs.rows() == rhs.rows() && lhs.columns() == rhs.columns());
    }

    T operator()(std::size_t row, std::size_t col) const override {
        return _lhs(row, col) * _rhs(row, col);
    }

    std::size_t rows() const override { return _lhs.rows(); }
    std::size_t columns() const override { return _lhs.columns(); }

private:
    const MatrixExpression<T>& _lhs;
    const MatrixExpression<T>& _rhs;
};

template <typename T>
class MatrixCwiseQuotient : public MatrixExpression<T> {
public:
    MatrixCwiseQuotient(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs)
        : _lhs(lhs), _rhs(rhs) {
        assert(lhs.rows() == rhs.rows() && lhs.columns() == rhs.columns());
    }

    T operator()(std::size_t row, std::size_t col) const override {
        return _lhs(row, col) / _rhs(row, col);
    }

    std::size_t rows() const override { return _lhs.rows(); }
    std::size_t columns() const override { return _lhs.columns(); }

private:
    const MatrixExpression<T>& _lhs;
    const MatrixExpression<T>& _rhs;
};

template <typename T>
class MatrixRowView : public MatrixExpression<T> {
public:
    MatrixRowView(const MatrixExpression<T>& mat, std::size_t row)
        : _mat(mat), _row(row) {
        assert(row < mat.rows());
    }

    T operator()(std::size_t row, std::size_t col) const override {
        assert(row == 0);
        return _mat(_row, col);
    }

    std::size_t rows() const override { return 1; }
    std::size_t columns() const override { return _mat.columns(); }

private:
    const MatrixExpression<T>& _mat;
    std::size_t _row;
};

template <typename T>
class MatrixColView : public MatrixExpression<T> {
public:
    MatrixColView(const MatrixExpression<T>& mat, std::size_t col)
        : _mat(mat), _col(col) {
        assert(col < mat.columns());
    }

    T operator()(std::size_t row, std::size_t col) const override {
        assert(col == 0);
        return _mat(row, _col);
    }

    std::size_t rows() const override { return _mat.rows(); }
    std::size_t columns() const override { return 1; }

private:
    const MatrixExpression<T>& _mat;
    std::size_t _col;
};

template <typename T>
class MatrixBlockView : public MatrixExpression<T> {
public:
    MatrixBlockView(const MatrixExpression<T>& mat, std::size_t startRow, std::size_t startCol,
                   std::size_t numRows, std::size_t numCols)
        : _mat(mat), _startRow(startRow), _startCol(startCol), _numRows(numRows), _numCols(numCols) {
        assert(startRow + numRows <= mat.rows() && startCol + numCols <= mat.columns());
    }

    T operator()(std::size_t row, std::size_t col) const override {
        assert(row < _numRows && col < _numCols);
        return _mat(_startRow + row, _startCol + col);
    }

    std::size_t rows() const override { return _numRows; }
    std::size_t columns() const override { return _numCols; }

private:
    const MatrixExpression<T>& _mat;
    std::size_t _startRow, _startCol, _numRows, _numCols;
};

template<typename T>
class Matrix : public MatrixExpression<T> {
public:
    /*----------------- Constructors ------------------*/
    Matrix() : _data(nullptr), _rows(0), _cols(0), _capacity(0) {}

    Matrix(std::size_t rows, std::size_t cols)
        : _data(new T[rows * cols]), _rows(rows), _cols(cols), _capacity(rows*cols) {}

    //copy constructor
    Matrix(const Matrix& other)
        : _data(new T[other._capacity]), _rows(other._rows), _cols(other._cols), _capacity(other._capacity) {
        cuda_matrix_copy(other.data(), _data, _capacity);
    }

    //move constructor
    Matrix(Matrix&& other) noexcept :
        _data(other._data), _rows(other._rows), _cols(other._cols), _capacity(other._capacity) {
        other._data = nullptr;
        other._rows = other._cols = other._capacity = 0;
    }

    //expression constructor
    Matrix(const MatrixExpression<T>& expr) : _rows(expr.rows()), _cols(expr.columns()), _capacity(_rows * _cols) {
        assert(_rows > 0 && _cols > 0);
        _data = new T[_capacity];

        for (std::size_t i = 0; i < _rows; ++i) {
            for (std::size_t j = 0; j < _cols; ++j) {
                (*this)(i, j) = expr(i, j);
            }
        }
    }

    //copy assignment
    Matrix& operator=(const Matrix& other) {
        if(this != &other) {
            delete[] _data;
            _rows = other._rows;
            _cols = other._cols;
            _capacity = other._capacity;
            _data = new T[_capacity];
            cuda_matrix_copy(other.data(), _data, _capacity);
        }
        return *this;
    }

    //move assignment
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] _data;
            _rows = other._rows;
            _cols = other._cols;
            _capacity = other._capacity;
            _data = other._data;
            other._data = nullptr;
            other._rows = other._cols = other._capacity = 0;
        }
        return *this;
    }

    //destructor
    ~Matrix() {
        delete[] _data;
    }

    /*----------------- Accessors ------------------*/
    T operator()(std::size_t row, std::size_t col) const override {
        return _data[row * _cols + col];
    }

    T& operator()(std::size_t row, std::size_t col) {
        return _data[row * _cols + col];
    }

    std::size_t rows() const override { return _rows; }
    std::size_t columns() const override { return _cols; }

    T* data() { return _data; }
    const T* data() const { return _data; }

    /*-----------------Modifiers--------------*/
    void setConstant(const T& value) {
        std::fill(_data, _data + _capacity, value);
    }

    void setZero() {
        static_assert(std::is_arithmetic<T>::value, "setZero() requires an arithmetic type.");
        cuda_matrix_clear(_data, _capacity);
    }

    void setOnes() {
        static_assert(std::is_arithmetic<T>::value, "setOnes() requires an arithmetic type.");
        std::fill(_data, _data + _capacity, static_cast<T>(1));
    }

    void setRandom() {
        static_assert(std::is_arithmetic<T>::value, "setRandom() requires an arithmetic type.");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for(std::size_t i = 0; i < _capacity; ++i) {
            if constexpr (std::is_integral<T>::value) {
                _data[i] = static_cast<T>(dis(gen) * 100); //random int 0-99
            } else {
                _data[i] = static_cast<T>(dis(gen)); // random float/double
            }
        }
    }

    /*----------------Iterator----------------*/
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using reference         = T&;

        Iterator(T* p) : _curr(p) {}

        reference operator*() const {
            return *_curr;
        }

        Iterator& operator++() {
            ++_curr;
            return *this;
        }

        Iterator& operator--() {
            --_curr;
            return *this;
        }

        Iterator operator+(int n) const {
            return Iterator(_curr + n);
        }

        difference_type operator-(const Iterator& other) const {
            return _curr - other._curr;
        }

        Iterator& operator+=(int n) {
            _curr += n;
            return *this;
        }

        bool operator==(const Iterator& other) const {
            return _curr == other._curr;
        }

        bool operator!=(const Iterator& other) const {
            return _curr != other._curr;
        }

    private:
        T* _curr;
    };

    Iterator begin() {
        return Iterator(_data);
    }

    Iterator end() {
        return Iterator(_data + _capacity);
    }

    const Iterator begin() const {
        return Iterator(_data);
    }

    const Iterator end() const {
        return Iterator(_data + _capacity);
    }

    /*--------------Operations----------------*/
    Matrix<T> transpose() const {
        Matrix<T> result(_cols, _rows);
        cuda_matrix_transpose(_data, result.data(), _rows, _cols);
        return result;
    }

    Matrix<T> cwiseProduct(const Matrix<T>& other) const {
        assert(_rows == other._rows && _cols == other._cols);
        Matrix<T> result(*this);
        cuda_matrix_op(result.data(), other.data(), _capacity, MultiplyOp<T>());
        return result;
    }

    Matrix<T> cwiseQuotient(const Matrix<T>& other) const {
        assert(_rows == other._rows && _cols == other._cols);
        Matrix<T> result(*this);
        cuda_matrix_op(result.data(), other.data(), _capacity, DivideOp<T>());
        return result;
    }

    T dot(const Matrix<T>& other) const {
        assert(_rows == other._rows && _cols == other._cols);
        T* result = new T(0);
        cuda_matrix_dot(_data, other.data(), _capacity, result);
        T value = *result;
        delete result;
        return value;
    }

    T sum() const {
        return std::accumulate(_data, _data + _capacity, T());
    }

    T minCoeff() const {
        assert(_capacity > 0);
        return *std::min_element(_data, _data + _capacity);
    }

    T maxCoeff() const {
        assert(_capacity > 0);
        return *std::max_element(_data, _data + _capacity);
    }

    T norm() const {
        return std::sqrt(this->dot(*this));
    }

    Matrix<T> normalized() const {
        T n = norm();
        assert(n != static_cast<T>(0) && "Cannot normalize a matrix with zero norm.");
        Matrix<T> result(*this);
        cuda_matrix_scalar_op(result.data(), static_cast<T>(1)/n, _capacity, MultiplyOp<T>());
        return result;
    }

    MatrixRowView<T> row(std::size_t i) const {
        return MatrixRowView<T>(*this, i);
    }

    MatrixColView<T> col(std::size_t j) const {
        return MatrixColView<T>(*this, j);
    }

    MatrixBlockView<T> block(std::size_t row, std::size_t col, std::size_t r, std::size_t c) const {
        return MatrixBlockView<T>(*this, row, col, r, c);
    }

    void resize(std::size_t new_rows, std::size_t new_cols) {
        std::size_t new_capacity = new_rows * new_cols;

        if (new_capacity != _capacity) {
            T* new_data = new T[new_capacity];
            std::size_t min_rows = std::min(_rows, new_rows);
            std::size_t min_cols = std::min(_cols, new_cols);

            for (std::size_t i = 0; i < min_rows; ++i) {
                std::copy_n(
                    _data + i * _cols,
                    min_cols,
                    new_data + i * new_cols
                );
            }

            delete[] _data;
            _data = new_data;
            _capacity = new_capacity;
        }

        _rows = new_rows;
        _cols = new_cols;
    }

    // Matrix multiplication
    Matrix<T> operator*(const Matrix<T>& other) const {
        assert(_cols == other._rows && "Matrix dimensions mismatch for multiplication");
        Matrix<T> result(_rows, other._cols);
        cuda_matrix_multiply(_data, other.data(), result.data(), _rows, _cols, other._cols);
        return result;
    }

    // Element-wise addition
    Matrix<T>& operator+=(const Matrix<T>& other) {
        assert(_rows == other._rows && _cols == other._cols && "Matrix dimensions mismatch");
        cuda_matrix_op(_data, other.data(), _capacity, AddOp<T>());
        return *this;
    }

    // Element-wise subtraction
    Matrix<T>& operator-=(const Matrix<T>& other) {
        assert(_rows == other._rows && _cols == other._cols && "Matrix dimensions mismatch");
        cuda_matrix_op(_data, other.data(), _capacity, SubtractOp<T>());
        return *this;
    }

    // Scalar multiplication
    Matrix<T>& operator*=(const T& scalar) {
        cuda_matrix_scalar_op(_data, scalar, _capacity, MultiplyOp<T>());
        return *this;
    }

    // Scalar division
    Matrix<T>& operator/=(const T& scalar) {
        assert(scalar != static_cast<T>(0) && "Division by zero");
        cuda_matrix_scalar_op(_data, scalar, _capacity, DivideOp<T>());
        return *this;
    }

private:
    T* _data;
    std::size_t _rows;
    std::size_t _cols;
    std::size_t _capacity; // _rows * _cols
};

// Global Operators
template<typename T>
Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b) {
    assert(a.rows() == b.rows() && a.columns() == b.columns() && "Matrix dimensions mismatch");
    Matrix<T> result(a);
    result += b;
    return result;
}

template<typename T>
Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b) {
    assert(a.rows() == b.rows() && a.columns() == b.columns() && "Matrix dimensions mismatch");
    Matrix<T> result(a);
    result -= b;
    return result;
}

template<typename T>
Matrix<T> operator*(const Matrix<T>& a, const T& scalar) {
    Matrix<T> result(a);
    result *= scalar;
    return result;
}

template<typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& a) {
    return a * scalar;
}

template<typename T>
Matrix<T> operator/(const Matrix<T>& a, const T& scalar) {
    assert(scalar != static_cast<T>(0) && "Division by zero");
    Matrix<T> result(a);
    result /= scalar;
    return result;
}

template<typename T>
bool operator==(const Matrix<T>& a, const Matrix<T>& b) {
    if (a.rows() != b.rows() || a.columns() != b.columns()) return false;
    bool* equality = new bool(true);
    cuda_matrix_compare_equality(a.data(), b.data(), a.rows() * a.columns(), equality);
    bool result = *equality;
    delete equality;
    return result;
}

template<typename T>
bool operator!=(const Matrix<T>& a, const Matrix<T>& b) {
    return !(a == b);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
    for(std::size_t i = 0; i < matrix.rows(); ++i) {
        for(std::size_t j = 0; j < matrix.columns(); ++j) {
            os << matrix(i, j) << " ";
        }
        os << '\n';
    }
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const MatrixExpression<T>& expr) {
    for (std::size_t i = 0; i < expr.rows(); ++i) {
        for (std::size_t j = 0; j < expr.columns(); ++j) {
            os << expr(i, j) << " ";
        }
        os << '\n';
    }
    return os;
}

template<typename T>
class MatrixMul : public MatrixExpression<T> {
public:
    MatrixMul(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs) : _lhs(lhs), _rhs(rhs) {
        assert(lhs.columns() == rhs.rows() && "Matrix dimensions incompatible for multiplication.");
    }

    T operator()(std::size_t row, std::size_t col) const override {
        T sum = T();  // zero-initialize
        for (std::size_t k = 0; k < _lhs.columns(); ++k) {
            sum += _lhs(row, k) * _rhs(k, col);
        }
        return sum;
    }

    std::size_t rows() const override { return _lhs.rows(); }
    std::size_t columns() const override { return _rhs.columns(); }

private:
    const MatrixExpression<T>& _lhs;
    const MatrixExpression<T>& _rhs;
};

template<typename T>
MatrixMul<T> operator*(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs) {
    return MatrixMul<T>(lhs, rhs);
}

template<typename T>
class MatrixAdd : public MatrixExpression<T> {
public:
    MatrixAdd(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs) : _lhs(lhs), _rhs(rhs) {
        assert(lhs.rows() == rhs.rows() && lhs.columns() == rhs.columns());
    }

    T operator()(std::size_t row, std::size_t col) const override {
        return _lhs(row, col) + _rhs(row, col);
    }

    std::size_t rows() const override {
        return _lhs.rows();
    }

    std::size_t columns() const override {
        return _lhs.columns();
    }

private:
    const MatrixExpression<T>& _lhs;
    const MatrixExpression<T>& _rhs;
};

template<typename T>
MatrixAdd<T> operator+(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs) {
    return MatrixAdd<T>(lhs, rhs);
}

template<typename T>
class MatrixSub : public MatrixExpression<T> {
public:
    MatrixSub(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs) : _lhs(lhs), _rhs(rhs) {
        assert(lhs.rows() == rhs.rows() && lhs.columns() == rhs.columns());
    }

    T operator()(std::size_t row, std::size_t col) const override {
        return _lhs(row, col) - _rhs(row, col);
    }

    std::size_t rows() const override {
        return _lhs.rows();
    }

    std::size_t columns() const override {
        return _lhs.columns();
    }

private:
    const MatrixExpression<T>& _lhs;
    const MatrixExpression<T>& _rhs;
};

template<typename T>
MatrixSub<T> operator-(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs) {
    return MatrixSub<T>(lhs, rhs);
}
// Factory functions
template <typename T>
Matrix<T> Identity(std::size_t n) {
    Matrix<T> result(n, n);
    result.setZero();
    for (std::size_t i = 0; i < n; ++i) {
        result(i, i) = static_cast<T>(1);
    }
    return result;
}

template <typename T>
Matrix<T> Ones(std::size_t rows, std::size_t cols) {
    Matrix<T> result(rows, cols);
    result.setOnes();
    return result;
}

template <typename T>
Matrix<T> Zeros(std::size_t rows, std::size_t cols) {
    Matrix<T> result(rows, cols);
    result.setZero();
    return result;
}

template <typename T>
Matrix<T> Random(std::size_t rows, std::size_t cols) {
    Matrix<T> result(rows, cols);
    result.setRandom();
    return result;
}

// Additional utility functions
template <typename T>
Matrix<T> hstack(const Matrix<T>& a, const Matrix<T>& b) {
    assert(a.rows() == b.rows() && "Matrices must have the same number of rows for horizontal stack");
    Matrix<T> result(a.rows(), a.columns() + b.columns());

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.columns(); ++j) {
            result(i, j) = a(i, j);
        }
        for (std::size_t j = 0; j < b.columns(); ++j) {
            result(i, a.columns() + j) = b(i, j);
        }
    }

    return result;
}

template <typename T>
Matrix<T> vstack(const Matrix<T>& a, const Matrix<T>& b) {
    assert(a.columns() == b.columns() && "Matrices must have the same number of columns for vertical stack");
    Matrix<T> result(a.rows() + b.rows(), a.columns());

    for (std::size_t j = 0; j < a.columns(); ++j) {
        for (std::size_t i = 0; i < a.rows(); ++i) {
            result(i, j) = a(i, j);
        }
        for (std::size_t i = 0; i < b.rows(); ++i) {
            result(a.rows() + i, j) = b(i, j);
        }
    }

    return result;
}

// Solvers (basic implementation)
template <typename T>
Matrix<T> solveLinearSystem(const Matrix<T>& A, const Matrix<T>& b) {
    // This is a simplified implementation using Gaussian elimination
    // For production code, consider using more stable methods or libraries like LAPACK
    assert(A.rows() == A.columns() && "Matrix A must be square");
    assert(A.rows() == b.rows() && b.columns() == 1 && "Dimensions mismatch for Ax = b");

    size_t n = A.rows();
    Matrix<T> augmented = hstack(A, b);

    // Forward elimination
    for (size_t i = 0; i < n; ++i) {
        // Find pivot
        size_t pivotRow = i;
        T pivotVal = std::abs(augmented(i, i));

        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(augmented(j, i)) > pivotVal) {
                pivotVal = std::abs(augmented(j, i));
                pivotRow = j;
            }
        }

        // Check for singular matrix
        if (pivotVal < std::numeric_limits<T>::epsilon()) {
            throw std::runtime_error("Matrix is singular or nearly singular");
        }

        // Swap rows if needed
        if (pivotRow != i) {
            for (size_t j = i; j <= n; ++j) {
                std::swap(augmented(i, j), augmented(pivotRow, j));
            }
        }

        // Eliminate below
        for (size_t j = i + 1; j < n; ++j) {
            T factor = augmented(j, i) / augmented(i, i);
            for (size_t k = i; k <= n; ++k) {
                augmented(j, k) -= factor * augmented(i, k);
            }
        }
    }

    // Back substitution
    Matrix<T> x(n, 1);
    for (int i = n - 1; i >= 0; --i) {
        T sum = T();
        for (size_t j = i + 1; j < n; ++j) {
            sum += augmented(i, j) * x(j, 0);
        }
        x(i, 0) = (augmented(i, n) - sum) / augmented(i, i);
    }

    return x;
}



#endif //MATRIXCUDA_CUH
