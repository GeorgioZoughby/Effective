#ifndef MATRIXEXPRESSION_H
#define MATRIXEXPRESSION_H

#include <cstddef> // for size_t

template<typename T>
class MatrixExpression {
public:
    virtual T operator()(std::size_t row, std::size_t col) const = 0;
    virtual std::size_t rows() const = 0;
    virtual std::size_t columns() const = 0;
    virtual ~MatrixExpression() = default;
};

#endif // MATRIXEXPRESSION_H
