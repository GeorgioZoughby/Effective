#ifndef MATRIXCWISEPRODUCT_H
#define MATRIXCWISEPRODUCT_H

#include "MatrixExpression.h"
#include <cassert>

template<typename T>
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

#endif // MATRIXCWISEPRODUCT_H
