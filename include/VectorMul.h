#include <cstddef>
#include "VectorExpression.h"

#ifndef VECTORMUL_H
#define VECTORMUL_H

namespace expr {

    template<typename T>
    class VectorMul : public VectorExpression<T> {
    public:
        VectorMul(const VectorExpression<T>& lhs , const VectorExpression<T>& rhs) : _lhs(lhs), _rhs(rhs) {}

        T operator[](size_t i) const override {
            return _lhs[i] * _rhs[i];
        }

        size_t size() const override {
            return _lhs.size();
        }

    private:
        const VectorExpression<T>& _lhs;
        const VectorExpression<T>& _rhs;
    };

    template <typename T>
 VectorMul<T> operator*(const VectorExpression<T>& a, const VectorExpression<T>& b) {
        return VectorMul<T>(a, b);
    }


}

#endif // VECTORMUL_H
