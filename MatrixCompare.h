#ifndef MATRIXCOMPARE_H
#define MATRIXCOMPARE_H

#include "MatrixExpression.h"
#include "Matrix.h"
#include <cassert>
#include <cstddef>

// Matrix == Matrix
template<typename T>
bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns())
        return false;

    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t j = 0; j < lhs.columns(); ++j) {
            if (lhs(i, j) != rhs(i, j)) {
                return false;
            }
        }
    }
    return true;
}

// Matrix == Expression
template<typename T>
bool operator==(const Matrix<T>& lhs, const MatrixExpression<T>& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns()) {
        return false;
    }

    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t j = 0; j < lhs.columns(); ++j) {
            if (lhs(i, j) != rhs(i, j)) {
                return false;
            }
        }
    }
    return true;
}

// Expression == Matrix
template<typename T>
bool operator==(const MatrixExpression<T>& lhs, const Matrix<T>& rhs) {
    return rhs == lhs; 
}

// Matrix != Matrix
template<typename T>
bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    return !(lhs == rhs);
}

// Matrix != Expression
template<typename T>
bool operator!=(const Matrix<T>& lhs, const MatrixExpression<T>& rhs) {
    return !(lhs == rhs);
}

// Expression != Matrix
template<typename T>
bool operator!=(const MatrixExpression<T>& lhs, const Matrix<T>& rhs) {
    return !(lhs == rhs);
}

// Expression == Expression
template<typename T>
bool operator==(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns()) {
        return false;
    }

    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t j = 0; j < lhs.columns(); ++j) {
            if (lhs(i, j) != rhs(i, j)) {
                return false;
            }
        }
    }
    return true;
}

// Expression != Expression
template<typename T>
bool operator!=(const MatrixExpression<T>& lhs, const MatrixExpression<T>& rhs) {
    return !(lhs == rhs);
}

#endif // MATRIXCOMPARE_H
