#ifndef MATRIXROWVIEW_H
#define MATRIXROWVIEW_H

#include "MatrixExpression.h"
#include <cassert>

template<typename T>
class MatrixRowView : public MatrixExpression<T> {
public:
    MatrixRowView(const MatrixExpression<T>& mat, std::size_t row)
        : _matrix(mat), _row(row) {
        assert(row < mat.rows());
    }

    T operator()(std::size_t i, std::size_t j) const override {
        assert(i == 0 && "Row view has only 1 row — index must be 0");
        return _matrix(_row, j);
    }

    std::size_t rows() const override { return 1; }
    std::size_t columns() const override { return _matrix.columns(); }

private:
    const MatrixExpression<T>& _matrix;
    std::size_t _row;
};

#endif // MATRIXROWVIEW_H
