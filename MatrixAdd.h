#ifndef MATRIXADD_H
#define MATRIXADD_H

#include "MatrixExpression.h"
#include <cassert>

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

#endif // MATRIXADD_H