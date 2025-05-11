#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <algorithm>
#include <random>
#include <cassert>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <immintrin.h> // Added for AVX intrinsics

#include "MatrixExpression.h"
#include "MatrixTranspose.h"
#include "MatrixCwiseProduct.h"
#include "MatrixCwiseQuotient.h"
#include "MatrixRowView.h"
#include "MatrixColView.h"
#include "MatrixBlockView.h"

template <typename T>
class Matrix : public MatrixExpression<T>
{
public:
    /*----------------- Constructors ------------------*/
    Matrix() : _data(nullptr), _rows(0), _cols(0), _capacity(0) {}

    Matrix(std::size_t rows, std::size_t cols)
        : _data(new T[rows * cols]), _rows(rows), _cols(cols), _capacity(rows * cols) {}

    // copy constructor
    Matrix(const Matrix &other)
        : _data(new T[other._capacity]), _rows(other._rows), _cols(other._cols), _capacity(other._capacity)
    {
        std::copy(other.begin(), other.end(), _data);
    }

    // move constructor
    Matrix(Matrix &&other) noexcept : _data(other._data), _rows(other._rows), _cols(other._cols), _capacity(other._capacity)
    {
        other._data = nullptr;
        other._rows = other._cols = other._capacity = 0;
    }

    // expression constructor
    Matrix(const MatrixExpression<T> &expr) : _rows(expr.rows()), _cols(expr.columns()), _capacity(_rows * _cols)
    {
        assert(_rows > 0 && _cols > 0);
        _data = new T[_capacity];

        for (std::size_t i = 0; i < _rows; ++i)
        {
            for (std::size_t j = 0; j < _cols; ++j)
            {
                (*this)(i, j) = expr(i, j);
            }
        }
    }

    // copy assignment
    Matrix &operator=(const Matrix &other)
    {
        if (this != &other)
        {
            delete[] _data;
            _rows = other._rows;
            _cols = other._cols;
            _capacity = other._capacity;
            _data = new T[_capacity];
            std::copy(other.begin(), other.end(), _data);
        }
        return *this;
    }

    // move assignment
    Matrix &operator=(Matrix &&other) noexcept
    {
        if (this != &other)
        {
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

    // destructor
    ~Matrix()
    {
        delete[] _data;
    }

    /*----------------------------------------*/

    /*----------------- Accessors ------------------*/

    T operator()(std::size_t row, std::size_t col) const override
    {
        return _data[row * _cols + col];
    }

    T &operator()(std::size_t row, std::size_t col)
    {
        return _data[row * _cols + col];
    }

    std::size_t rows() const override { return _rows; }
    std::size_t columns() const override { return _cols; }

    T *data() { return _data; }
    const T *data() const { return _data; }

    /*----------------------------------------*/

    /*-----------------Modifiers--------------*/

    void setConstant(const T &value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m256 val_vec = _mm256_set1_ps(value);

#pragma omp parallel for
            for (std::size_t i = 0; i + 8 <= _capacity; i += 8)
            {
                _mm256_storeu_ps(&_data[i], val_vec);
            }

#pragma omp parallel for
            for (std::size_t i = (_capacity / 8) * 8; i < _capacity; ++i)
            {
                _data[i] = value;
            }
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m256d val_vec = _mm256_set1_pd(value);

#pragma omp parallel for
            for (std::size_t i = 0; i + 4 <= _capacity; i += 4)
            {
                _mm256_storeu_pd(&_data[i], val_vec);
            }

#pragma omp parallel for
            for (std::size_t i = (_capacity / 4) * 4; i < _capacity; ++i)
            {
                _data[i] = value;
            }
        }
        else
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < _capacity; ++i)
            {
                _data[i] = value;
            }
        }
    }

    void setZero()
    {
        static_assert(std::is_arithmetic<T>::value, "setZero() requires an arithmetic type.");

        if constexpr (std::is_same_v<T, float>)
        {
            __m256 zero_vec = _mm256_setzero_ps();

#pragma omp parallel for
            for (std::size_t i = 0; i + 8 <= _capacity; i += 8)
            {
                _mm256_storeu_ps(&_data[i], zero_vec);
            }

#pragma omp parallel for
            for (std::size_t i = (_capacity / 8) * 8; i < _capacity; ++i)
            {
                _data[i] = 0.0f;
            }
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m256d zero_vec = _mm256_setzero_pd();

#pragma omp parallel for
            for (std::size_t i = 0; i + 4 <= _capacity; i += 4)
            {
                _mm256_storeu_pd(&_data[i], zero_vec);
            }

#pragma omp parallel for
            for (std::size_t i = (_capacity / 4) * 4; i < _capacity; ++i)
            {
                _data[i] = 0.0;
            }
        }
        else
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < _capacity; ++i)
            {
                _data[i] = static_cast<T>(0);
            }
        }
    }

    void setOnes()
    {
        static_assert(std::is_arithmetic<T>::value, "setOnes() requires an arithmetic type.");

        if constexpr (std::is_same_v<T, float>)
        {
            __m256 ones_vec = _mm256_set1_ps(1.0f);

#pragma omp parallel for
            for (std::size_t i = 0; i + 8 <= _capacity; i += 8)
            {
                _mm256_storeu_ps(&_data[i], ones_vec);
            }

#pragma omp parallel for
            for (std::size_t i = (_capacity / 8) * 8; i < _capacity; ++i)
            {
                _data[i] = 1.0f;
            }
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m256d ones_vec = _mm256_set1_pd(1.0);

#pragma omp parallel for
            for (std::size_t i = 0; i + 4 <= _capacity; i += 4)
            {
                _mm256_storeu_pd(&_data[i], ones_vec);
            }

#pragma omp parallel for
            for (std::size_t i = (_capacity / 4) * 4; i < _capacity; ++i)
            {
                _data[i] = 1.0;
            }
        }
        else
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < _capacity; ++i)
            {
                _data[i] = static_cast<T>(1);
            }
        }
    }

    void setRandom()
    {
        static_assert(std::is_arithmetic<T>::value, "setRandom() requires an arithmetic type.");
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

#pragma omp parallel for
        for (std::size_t i = 0; i < _capacity; ++i)
        {
            if constexpr (std::is_integral<T>::value)
            {
                _data[i] = static_cast<T>(dis(gen) * 100); // random int 0-99
            }
            else
            {
                _data[i] = static_cast<T>(dis(gen)); // random float/double
            }
        }
    }
    /*----------------------------------------*/

    /*----------------Iterator----------------*/
    class Iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T *;
        using reference = T &;

        Iterator(T *p) : _curr(p) {}

        reference operator*() const
        {
            return *_curr;
        }

        Iterator &operator++()
        {
            ++_curr;
            return *this;
        }

        Iterator &operator--()
        {
            --_curr;
            return *this;
        }

        Iterator operator+(int n) const
        {
            return Iterator(_curr + n);
        }

        difference_type operator-(const Iterator &other) const
        {
            return _curr - other._curr;
        }

        Iterator &operator+=(int n)
        {
            _curr += n;
            return *this;
        }

        bool operator==(const Iterator &other) const
        {
            return _curr == other._curr;
        }

        bool operator!=(const Iterator &other) const
        {
            return _curr != other._curr;
        }

    private:
        T *_curr;
    };

    Iterator begin()
    {
        return Iterator(_data);
    }

    Iterator end()
    {
        return Iterator(_data + _capacity);
    }

    const Iterator begin() const
    {
        return Iterator(_data);
    }

    const Iterator end() const
    {
        return Iterator(_data + _capacity);
    }

    /*----------------------------------------*/

    /*--------------Operations----------------*/
    MatrixTranspose<T> transpose() const
    {
        return MatrixTranspose<T>(*this);
    }

    MatrixCwiseProduct<T> cwiseProduct(const MatrixExpression<T> &other) const
    {
        return MatrixCwiseProduct<T>(*this, other);
    }

    MatrixCwiseQuotient<T> cwiseQuotient(const MatrixExpression<T> &other) const
    {
        return MatrixCwiseQuotient<T>(*this, other);
    }

    T dot(const MatrixExpression<T> &other) const
    {
        assert(_rows == other.rows() && _cols == other.columns());

        if constexpr (std::is_same_v<T, float>)
        {
            float result = 0.0f;

#pragma omp parallel
            {
                __m256 local_sum = _mm256_setzero_ps();

#pragma omp for collapse(2) nowait
                for (std::size_t i = 0; i < _rows; ++i)
                {
                    for (std::size_t j = 0; j + 8 <= _cols; j += 8)
                    {
                        // For MatrixExpression, we need to load values one by one
                        float a_values[8];
                        float b_values[8];

                        for (int k = 0; k < 8; ++k)
                        {
                            a_values[k] = _data[i * _cols + j + k];
                            b_values[k] = other(i, j + k);
                        }

                        // Load into AVX registers
                        __m256 a = _mm256_loadu_ps(a_values);
                        __m256 b = _mm256_loadu_ps(b_values);

                        // Compute a * b + local_sum
                        local_sum = _mm256_fmadd_ps(a, b, local_sum);
                    }
                }

                // Process remaining elements
                float temp[8];
                _mm256_storeu_ps(temp, local_sum);
                float thread_sum = temp[0] + temp[1] + temp[2] + temp[3] +
                                   temp[4] + temp[5] + temp[6] + temp[7];

#pragma omp for collapse(2) reduction(+ : thread_sum)
                for (std::size_t i = 0; i < _rows; ++i)
                {
                    for (std::size_t j = (_cols / 8) * 8; j < _cols; ++j)
                    {
                        thread_sum += (*this)(i, j) * other(i, j);
                    }
                }

#pragma omp atomic
                result += thread_sum;
            }

            return result;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            double result = 0.0;

#pragma omp parallel
            {
                __m256d local_sum = _mm256_setzero_pd();

#pragma omp for collapse(2) nowait
                for (std::size_t i = 0; i < _rows; ++i)
                {
                    for (std::size_t j = 0; j + 4 <= _cols; j += 4)
                    {
                        // For MatrixExpression, we need to load values one by one
                        double a_values[4];
                        double b_values[4];

                        for (int k = 0; k < 4; ++k)
                        {
                            a_values[k] = _data[i * _cols + j + k];
                            b_values[k] = other(i, j + k);
                        }

                        // Load into AVX registers
                        __m256d a = _mm256_loadu_pd(a_values);
                        __m256d b = _mm256_loadu_pd(b_values);

                        // Compute a * b + local_sum
                        local_sum = _mm256_fmadd_pd(a, b, local_sum);
                    }
                }

                // Process remaining elements
                double temp[4];
                _mm256_storeu_pd(temp, local_sum);
                double thread_sum = temp[0] + temp[1] + temp[2] + temp[3];

#pragma omp for collapse(2) reduction(+ : thread_sum)
                for (std::size_t i = 0; i < _rows; ++i)
                {
                    for (std::size_t j = (_cols / 4) * 4; j < _cols; ++j)
                    {
                        thread_sum += (*this)(i, j) * other(i, j);
                    }
                }

#pragma omp atomic
                result += thread_sum;
            }

            return result;
        }
        else
        {
            T sum = T();

#pragma omp parallel for collapse(2) reduction(+ : sum)
            for (std::size_t i = 0; i < _rows; ++i)
            {
                for (std::size_t j = 0; j < _cols; ++j)
                {
                    sum += (*this)(i, j) * other(i, j);
                }
            }

            return sum;
        }
    }

    T sum() const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            float result = 0.0f;

#pragma omp parallel
            {
                __m256 local_sum = _mm256_setzero_ps();

#pragma omp for nowait
                for (std::size_t i = 0; i + 8 <= _capacity; i += 8)
                {
                    __m256 data_vec = _mm256_loadu_ps(&_data[i]);
                    local_sum = _mm256_add_ps(local_sum, data_vec);
                }

                // Reduce the vector to a scalar
                float temp[8];
                _mm256_storeu_ps(temp, local_sum);
                float thread_sum = temp[0] + temp[1] + temp[2] + temp[3] +
                                   temp[4] + temp[5] + temp[6] + temp[7];

                // Process remaining elements
#pragma omp for reduction(+ : thread_sum)
                for (std::size_t i = (_capacity / 8) * 8; i < _capacity; ++i)
                {
                    thread_sum += _data[i];
                }

#pragma omp atomic
                result += thread_sum;
            }

            return result;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            double result = 0.0;

#pragma omp parallel
            {
                __m256d local_sum = _mm256_setzero_pd();

#pragma omp for nowait
                for (std::size_t i = 0; i + 4 <= _capacity; i += 4)
                {
                    __m256d data_vec = _mm256_loadu_pd(&_data[i]);
                    local_sum = _mm256_add_pd(local_sum, data_vec);
                }

                // Reduce the vector to a scalar
                double temp[4];
                _mm256_storeu_pd(temp, local_sum);
                double thread_sum = temp[0] + temp[1] + temp[2] + temp[3];

                // Process remaining elements
#pragma omp for reduction(+ : thread_sum)
                for (std::size_t i = (_capacity / 4) * 4; i < _capacity; ++i)
                {
                    thread_sum += _data[i];
                }

#pragma omp atomic
                result += thread_sum;
            }

            return result;
        }
        else
        {
            T total = T();

#pragma omp parallel for reduction(+ : total)
            for (std::size_t i = 0; i < _capacity; ++i)
            {
                total += _data[i];
            }

            return total;
        }
    }

    T minCoeff() const
    {
        assert(_capacity > 0);
        return *std::min_element(_data, _data + _capacity);
    }

    T maxCoeff() const
    {
        assert(_capacity > 0);
        return *std::max_element(_data, _data + _capacity);
    }

    T norm() const
    {
        return std::sqrt(this->dot(*this));
    }

    Matrix<T> normalized() const
    {
        T n = norm();
        assert(n != static_cast<T>(0) && "Cannot normalize a matrix with zero norm.");

        Matrix<T> result(_rows, _cols);

        if constexpr (std::is_same_v<T, float>)
        {
            __m256 norm_vec = _mm256_set1_ps(n);

#pragma omp parallel for
            for (std::size_t i = 0; i + 8 <= _capacity; i += 8)
            {
                __m256 data_vec = _mm256_loadu_ps(&_data[i]);
                __m256 result_vec = _mm256_div_ps(data_vec, norm_vec);
                _mm256_storeu_ps(&result._data[i], result_vec);
            }

#pragma omp parallel for
            for (std::size_t i = (_capacity / 8) * 8; i < _capacity; ++i)
            {
                result._data[i] = _data[i] / n;
            }
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m256d norm_vec = _mm256_set1_pd(n);

#pragma omp parallel for
            for (std::size_t i = 0; i + 4 <= _capacity; i += 4)
            {
                __m256d data_vec = _mm256_loadu_pd(&_data[i]);
                __m256d result_vec = _mm256_div_pd(data_vec, norm_vec);
                _mm256_storeu_pd(&result._data[i], result_vec);
            }

#pragma omp parallel for
            for (std::size_t i = (_capacity / 4) * 4; i < _capacity; ++i)
            {
                result._data[i] = _data[i] / n;
            }
        }
        else
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < _capacity; ++i)
            {
                result._data[i] = _data[i] / n;
            }
        }

        return result;
    }

    MatrixRowView<T> row(std::size_t i) const
    {
        return MatrixRowView<T>(*this, i);
    }

    MatrixColView<T> col(std::size_t j) const
    {
        return MatrixColView<T>(*this, j);
    }

    MatrixBlockView<T> block(std::size_t row, std::size_t col, std::size_t r, std::size_t c) const
    {
        return MatrixBlockView<T>(*this, row, col, r, c);
    }

    void resize(std::size_t new_rows, std::size_t new_cols)
    {
        std::size_t new_capacity = new_rows * new_cols;

        if (new_capacity != _capacity)
        {
            T *new_data = new T[new_capacity];
            std::size_t min_rows = std::min(_rows, new_rows);
            std::size_t min_cols = std::min(_cols, new_cols);

            for (std::size_t i = 0; i < min_rows; ++i)
            {
                std::copy_n(
                    _data + i * _cols,
                    min_cols,
                    new_data + i * new_cols);
            }

            delete[] _data;
            _data = new_data;
            _capacity = new_capacity;
        }

        _rows = new_rows;
        _cols = new_cols;
    }

    /*----------------------------------------*/

private:
    T *_data;
    std::size_t _rows;
    std::size_t _cols;
    std::size_t _capacity; //_rows*_columns
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix)
{
    for (std::size_t i = 0; i < matrix.rows(); ++i)
    {
        for (std::size_t j = 0; j < matrix.columns(); ++j)
        {
            os << matrix(i, j) << " ";
        }
        os << '\n';
    }
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const MatrixExpression<T> &expr)
{
    for (std::size_t i = 0; i < expr.rows(); ++i)
    {
        for (std::size_t j = 0; j < expr.columns(); ++j)
        {
            os << expr(i, j) << " ";
        }
        os << '\n';
    }
    return os;
}

#endif // MATRIX_H
