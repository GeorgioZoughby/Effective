#ifndef MATRIXCOLVIEW_H
#define MATRIXCOLVIEW_H

#include "MatrixExpression.h"
#include <cassert>

template<typename T>
class MatrixColView : public MatrixExpression<T> {
public:
    MatrixColView(const MatrixExpression<T>& mat, std::size_t col)
        : _matrix(mat), _col(col) {
        assert(col < mat.columns());
    }

    T operator()(std::size_t i, std::size_t j) const override {
        assert(j == 0 && "Column view has only 1 column — index must be 0");
        return _matrix(i, _col);
    }

    std::size_t rows() const override { return _matrix.rows(); }
    std::size_t columns() const override { return 1; }

private:
    const MatrixExpression<T>& _matrix;
    std::size_t _col;
};

#endif // MATRIXCOLVIEW_H
