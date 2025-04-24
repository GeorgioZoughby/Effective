#ifndef MATRIXMUL_H
#define MATRIXMUL_H

#include "MatrixExpression.h"
#include <cassert>

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

#endif // MATRIXMUL_H
