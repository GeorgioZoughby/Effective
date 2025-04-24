#ifndef MATRIXSCALARDIV_H
#define MATRIXSCALARDIV_H

#include "MatrixExpression.h"
#include <cassert>

template<typename T>
class MatrixScalarDiv : public MatrixExpression<T> {
public:
    MatrixScalarDiv(const MatrixExpression<T>& mat, T scalar)
        : _matrix(mat), _scalar(scalar) {
        assert(scalar != static_cast<T>(0) && "Division by zero!");
    }

    T operator()(std::size_t row, std::size_t col) const override {
        return _matrix(row, col) / _scalar;
    }

    std::size_t rows() const override { return _matrix.rows(); }
    std::size_t columns() const override { return _matrix.columns(); }

private:
    const MatrixExpression<T>& _matrix;
    T _scalar;
};

template<typename T>
MatrixScalarDiv<T> operator/(const MatrixExpression<T>& mat, T scalar) {
    return MatrixScalarDiv<T>(mat, scalar);
}

#endif // MATRIXSCALARDIV_H
