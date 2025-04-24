#ifndef MATRIXNEGATE_H
#define MATRIXNEGATE_H

#include "MatrixExpression.h"

template<typename T>
class MatrixNegate : public MatrixExpression<T> {
public:
    MatrixNegate(const MatrixExpression<T>& mat) : _matrix(mat) {}

    T operator()(std::size_t row, std::size_t col) const override {
        return -_matrix(row, col);
    }

    std::size_t rows() const override { return _matrix.rows(); }
    std::size_t columns() const override { return _matrix.columns(); }

private:
    const MatrixExpression<T>& _matrix;
};

template<typename T>
MatrixNegate<T> operator-(const MatrixExpression<T>& mat) {
    return MatrixNegate<T>(mat);
}

#endif // MATRIXNEGATE_H
