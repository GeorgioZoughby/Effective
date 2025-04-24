#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <algorithm>
#include <random>
#include <cassert>
#include <numeric>
#include <cmath>

#include "MatrixExpression.h"
#include "MatrixTranspose.h"
#include "MatrixCwiseProduct.h"
#include "MatricCwiseQuotient.h"
#include "MatrixRowView.h"
#include "MatrixColView.h"
#include "MatrixBlockView.h"


template<typename T>
class Matrix : public MatrixExpression<T>{
public:
        /*----------------- Constructors ------------------*/
        Matrix() : _data(nullptr) , _rows(0) , _cols(0) , _capacity(0) {}

        Matrix(std::size_t rows , std::size_t cols) 
        : _data( new T[rows * cols]) , _rows(rows) , _cols(cols) , _capacity(rows*cols) {}

        //copy constructor
        Matrix(const Matrix& other)
        : _data(new T[other._capacity]) , _rows(other._rows) , _cols(other._cols) , _capacity(other._capacity){
            std::copy(other.begin(), other.end() , _data);
        }

        //move constructor
        Matrix(Matrix&& other) noexcept : 
        _data(other._data) , _rows(other._rows) , _cols(other._cols) , _capacity(other._capacity) {
            other._data = nullptr;
            other._rows = other._cols = other._capacity = 0;
        }

        //expression constructor
        Matrix(const MatrixExpression<T>& expr) : _rows(expr.rows()), _cols(expr.columns()), _capacity(_rows * _cols)
        {
            assert(_rows > 0 && _cols > 0);
            _data = new T[_capacity];

            for (std::size_t i = 0; i < _rows; ++i) {
                for (std::size_t j = 0; j < _cols; ++j) {
                    (*this)(i, j) = expr(i, j);
                }
            }
        }

        //copy assignment
        Matrix& operator=(const Matrix& other){
            if(this != &other){
                delete[] _data;
                _rows = other._rows;
                _cols = other._cols;
                _capacity = other._capacity;
                _data = new T[_capacity];
                std::copy(other.begin() , other.end() , _data);
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

        /*----------------------------------------*/

        /*----------------- Accessors ------------------*/

        T operator()(std::size_t row , std::size_t col) const override {
            return _data[row * _cols + col];
        }

        T& operator()(std::size_t row , std::size_t col){
            return _data[row * _cols + col];
        }

        std::size_t rows() const override {return _rows;}
        std::size_t columns() const override {return _cols;}

        T* data() {return _data;}
        const T* data() const {return _data;}

        /*----------------------------------------*/

        /*-----------------Modifiers--------------*/

        void setConstant(const T& value){
            std::fill(_data , _data + _capacity , value);
        }

        void setZero() {
            static_assert(std::is_arithmetic<T>::value , "setZero() requires an arithmetic type.");
            std::fill(_data , _data + _capacity , static_cast<T>(0));
        }

        void setOnes() {
            static_assert(std::is_arithmetic<T>::value, "setOnes() requires an arithmetic type.");
            std::fill(_data, _data+_capacity , static_cast<T>(1));
        }

        void setRandom() {
            static_assert(std::is_arithmetic<T>::value, "setRandom() requires an arithmetic type.");
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0 , 1.0);

            for(std::size_t i = 0 ; i < _capacity ; ++i){
                if constexpr (std::is_integral<T>::value){
                    _data[i] = static_cast<T>(dis(gen) * 100); //random int 0-99
                } else {
                    _data[i] = static_cast<T>(dis(gen)); // random float/double
                }
            }
        }
        /*----------------------------------------*/

        /*----------------Iterator----------------*/
        class Iterator{
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

        /*----------------------------------------*/

        /*--------------Operations----------------*/
        MatrixTranspose<T> transpose() const {
            return MatrixTranspose<T>(*this);
        }

        MatrixCwiseProduct<T> cwiseProduct(const MatrixExpression<T>& other) const {
            return MatrixCwiseProduct<T>(*this,other);
        }

        MatrixCwiseQuotient<T> cwiseQuotient(const MatrixExpression<T>& other) const {
            return MatrixCwiseQuotient<T>(*this, other);
        }

        T dot(const MatrixExpression<T>& other) const {
            assert(_rows == other.rows() && _cols == other.columns());
            T sum = T();
            for (std::size_t i = 0; i < _rows; ++i) {
                for (std::size_t j = 0; j < _cols; ++j) {
                    sum += (*this)(i, j) * other(i, j);
                }
            }
            return sum;
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
            return *this / n;
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
        
        /*----------------------------------------*/


private:
    T* _data;
    std::size_t _rows;
    std::size_t _cols;
    std::size_t _capacity; //_rows*_columns

};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
    for(std::size_t i = 0 ; i < matrix.rows() ; ++i){
        for(std::size_t j = 0 ; j < matrix.columns() ; ++j){
            os << matrix(i,j) << " ";
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


#endif //MATRIX_H