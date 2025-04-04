#include <cstddef>
#include "VectorExpression.h"

#ifndef VECTORSUB_H
#define VECTORSUB_H

namespace expr {
    template<typename T>
    class VectorSub : public VectorExpression<T> {
    public:
        VectorSub(const VectorExpression<T> &lhs, const VectorExpression<T> &rhs) : _lhs(lhs), _rhs(rhs) {
        }

        T operator[](size_t i) const override {
            return _lhs[i] - _rhs[i];
        }

        size_t size() const override {
            return _lhs.size();
        }

    private:
        const VectorExpression<T> &_lhs;
        const VectorExpression<T> &_rhs;
    };

    template<typename T>
    VectorSub<T> operator-(const VectorExpression<T> &a, const VectorExpression<T> &b) {
        return VectorSub<T>(a, b);
    }
}


#endif //VECTORSUB_H
