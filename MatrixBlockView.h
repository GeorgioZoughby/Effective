#ifndef MATRIXBLOCKVIEW_H
#define MATRIXBLOCKVIEW_H

#include "MatrixExpression.h"
#include <cassert>

template<typename T>
class MatrixBlockView : public MatrixExpression<T> {
public:
    MatrixBlockView(const MatrixExpression<T>& mat, std::size_t row, std::size_t col, std::size_t rows, std::size_t cols)
        : _matrix(mat), _startRow(row), _startCol(col), _blockRows(rows), _blockCols(cols) {
        assert(row + rows <= mat.rows() && col + cols <= mat.columns());
    }

    T operator()(std::size_t i, std::size_t j) const override {
        return _matrix(_startRow + i, _startCol + j);
    }

    std::size_t rows() const override { return _blockRows; }
    std::size_t columns() const override { return _blockCols; }

private:
    const MatrixExpression<T>& _matrix;
    std::size_t _startRow, _startCol, _blockRows, _blockCols;
};

#endif // MATRIXBLOCKVIEW_H
