#ifndef MATRIXTRANSPOSE_H
#define MATRIXTRANSPOSE_H

#include "MatrixExpression.h"

template<typename T>
class MatrixTranspose : public MatrixExpression<T> {
public:
    MatrixTranspose(const MatrixExpression<T>& mat) : _matrix(mat) {}

    T operator()(std::size_t row, std::size_t col) const override {
        return _matrix(col, row); 
    }

    std::size_t rows() const override {
        return _matrix.columns();  
    }

    std::size_t columns() const override {
        return _matrix.rows();     
    }

private:
    const MatrixExpression<T>& _matrix;
};

#endif // MATRIXTRANSPOSE_H
