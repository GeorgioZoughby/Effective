#include <cstddef>
#include "VectorExpression.h"
#ifndef VECTORADD_H
#define VECTORADD_H

namespace expr{
	template<typename T>
    class VectorAdd : public VectorExpression<T> {
    public:
    	VectorAdd(const VectorExpression<T>& lhs , const VectorExpression<T>& rhs) : _lhs(lhs) , _rhs(rhs) {}
		T operator[](size_t i) const override {
        	return _lhs[i] + _rhs[i];
        }

		size_t size() const override {
			return _lhs.size();
        }



    private:
    	const VectorExpression<T>& _lhs;
        const VectorExpression<T>& _rhs;
    };

	template <typename T>
VectorAdd<T> operator+(const VectorExpression<T>& a, const VectorExpression<T>& b) {
		return VectorAdd<T>(a, b);
	}
}


#endif //VECTORADD_H
