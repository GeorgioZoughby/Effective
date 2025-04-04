#include <cstddef> // for size_t
#ifndef VECTOREXPRESSION_H
#define VECTOREXPRESSION_H

template<typename T>
class VectorExpression {
public:
    virtual T operator[](size_t i) const = 0;
    virtual size_t size() const = 0;
    virtual ~VectorExpression() = default;
};

#endif //VECTOREXPRESSION_H
