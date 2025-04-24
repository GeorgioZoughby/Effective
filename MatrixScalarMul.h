#ifndef MATRIXSCALARMUL_H
#define MATRIXSCALARMUL_H

#include "MatrixExpression.h"
#include <cassert>

template<typename T>
class MatrixScalarMul : public MatrixExpression<T> {
public:
    MatrixScalarMul(const MatrixExpression<T>& mat, T scalar)
        : _matrix(mat), _scalar(scalar) {}

    T operator()(std::size_t row, std::size_t col) const override {
        return _matrix(row, col) * _scalar;
    }

    std::size_t rows() const override {
        return _matrix.rows();
    }

    std::size_t columns() const override {
        return _matrix.columns();
    }

private:
    const MatrixExpression<T>& _matrix;
    T _scalar;
};

// Right scalar multiplication
template<typename T>
MatrixScalarMul<T> operator*(const MatrixExpression<T>& mat, T scalar) {
    return MatrixScalarMul<T>(mat, scalar);
}

// Left scalar multiplication
template<typename T>
MatrixScalarMul<T> operator*(T scalar, const MatrixExpression<T>& mat) {
    return MatrixScalarMul<T>(mat, scalar);
}

#endif // MATRIXSCALARMUL_H
