#ifndef VECTORCUDA_CUH
#define VECTORCUDA_CUH

#include <iostream>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <functional>
#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


template <typename T>
class Add {
public:
    __device__ void operator()(T& a, T b) const {
        a += b;
    }
};

template <typename T>
class Subtract {
public:
    __device__ void operator()(T& a, T b) const {
        a -= b;
    }
};

template <typename T>
class Multiply {
public:
    __device__ void operator()(T& a, T b) const {
        a *= b;
    }
};

template <typename T>
class Divide {
public:
    __device__ void operator()(T& a, T b) const {
        a /= b;
    }
};



//Expression Interface
template <typename T>
cudaError_t cuda_copy(const T* from, T* to, unsigned int size);
template <typename Op, typename T>
cudaError_t cuda_scalar_op(T* elts, const T scalar, unsigned int size, Op op);
template <typename T>
cudaError_t cuda_compare_equality(const T* elts1, const T* elts2, unsigned int size, bool* equality);
template <typename T>
cudaError_t cuda_dot(const T* elts1, const T* elts2, unsigned int size, T* result);
template <typename T>
cudaError_t cuda_clear(T* elts, unsigned int size);
template <typename Op, typename T>
cudaError_t cuda_vector_op(T* elts1, const T* elts2, unsigned int size, Op op);

template<typename T>
class VectorExpression {
public:
    virtual T operator[](size_t i) const = 0;
    virtual size_t size() const = 0;
    virtual ~VectorExpression() = default;
};

//Expression Implementation in a namespace
namespace expr {
    template<typename T>
    class VectorAdd : public VectorExpression<T> {
    public:
        VectorAdd(const VectorExpression<T>& lhs, const VectorExpression<T>& rhs) : _lhs(lhs), _rhs(rhs) {}

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

    template<typename T>
    class VectorSub : public VectorExpression<T> {
    public:

        VectorSub(const VectorExpression<T>& lhs, const VectorExpression<T>& rhs) : _lhs(lhs), _rhs(rhs) {}

        T operator[](size_t i) const override {
            return _lhs[i] - _rhs[i];
        }

        size_t size() const override {
            return _lhs.size();
        }

    private:
        const VectorExpression<T>& _lhs;
        const VectorExpression<T>& _rhs;
    };

    template<typename T>
    class VectorMul : public VectorExpression<T> {
    public:

        VectorMul(const VectorExpression<T>& lhs, const VectorExpression<T>& rhs) : _lhs(lhs), _rhs(rhs) {}

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
    VectorAdd<T> operator+(const VectorExpression<T>& a, const VectorExpression<T>& b) {
        return VectorAdd<T>(a, b);
    }


    template <typename T>
    VectorSub<T> operator-(const VectorExpression<T>& a, const VectorExpression<T>& b) {
        return VectorSub<T>(a, b);
    }


    template <typename T>
    VectorMul<T> operator*(const VectorExpression<T>& a, const VectorExpression<T>& b) {
        return VectorMul<T>(a, b);
    }
}



template <typename T>
class Vector : public VectorExpression<T> {
public:
    /*--------Constructors--------*/

    // Default Constructor
    Vector() : _size(0), _elements(nullptr), _capacity(0) {}

    // //Default initialisation of values--more safe--for large vector it will be slow
    // explicit Vector(int size): _size(size), _elements(new T[size]), _capacity(size){
    //     for(int i =0 ; i<_size ; i++){
    //         _elements[i] = T();
    //     }
    // }

    explicit Vector(int capacity) : _size(0), _elements(new T[capacity]), _capacity(capacity) {}

    // Copy Constructor
    Vector(const Vector<T>& other) : _size(other._size), _elements(new T[other._capacity]), _capacity(other._capacity) {
        cuda_copy(other._elements, _elements, _size);
    }

    // Move Constructor
    Vector(Vector<T>&& other) : _size(other._size), _elements(other._elements), _capacity(other._capacity) {
        other._size = 0;
        other._elements = nullptr;
        other._capacity = 0;
    }

    //Move Assignment Opeartor
    Vector<T>& operator=(Vector<T>&& other) noexcept {
        if (this != &other) {
            delete[] _elements;
            _size = other._size;
            _capacity = other._capacity;
            _elements = other._elements;

            other._size = 0;
            other._capacity = 0;
            other._elements = nullptr;
        }
        return *this;
    }

    //Copy Assignment Operator
    Vector<T>& operator=(const Vector<T>& other) {
        if (this != &other) {
            delete[] _elements;

            _size = other._size;
            _capacity = other._capacity;
            _elements = new T[_capacity];

            cuda_copy(_elements, other._elements, _size);
        }
        return *this;
    }

    // Destructor
    ~Vector() {
        delete[] _elements;
        _elements = nullptr;
        _size = 0;
        _capacity = 0;
    }

    /*----------------------------*/

    /*------Expression Vector-----*/

    //Expression Based Constructor
    Vector(const VectorExpression<T>& expr) {
        _size = expr.size();
        _capacity = _size;
        _elements = new T[_capacity];
        for (size_t i = 0; i < _size; i++) {
            _elements[i] = expr[i];
        }
    }

    T operator[](size_t i) const override {
        return _elements[i];
    }

    size_t size() const override {
        return _size;
    }

    /*-----------------------------*/

    /*--------Capacity--------*/

    size_t capacity() const {
        return _capacity;
    }

    bool empty() const {
        return _size == 0;
    }

    size_t max_size() const {
        return _capacity;
    }

    // Requests a change in capacity
    // reserve() will never decrase the capacity.
    void reserve(int newalloc) {
        if (newalloc <= 0) return;
        if (newalloc <= _capacity) return;

        T* tmp = new T[newalloc];

        std::move(begin(), end(), tmp);

        delete[] _elements;
        _elements = tmp;
        _capacity = newalloc;
    }

    //default initialisation and destroying the values can affect the performance to be conssidered
    // Changes the Vector's size.
    // If the newsize is smaller, the last elements will be destroyed.
    // Has a default value param for custom values when resizing.
    void resize(int newsize, T val = T()) {
        if (newsize < 0) return;
        if (newsize > _capacity) {
            reserve(newsize);
        }

        if (newsize > _size) {
            for (int i = _size; i < newsize; i++) {
                _elements[i] = val;
            }
        }
        else if (newsize < _size) {
            for (int i = newsize; i < _size; i++) {
                _elements[i].~T();
            }
        }

        _size = newsize;
    }
    /*----------------------------*/

    /*----------Modifiers---------*/

    // Removes all elements from the Vector
    // Capacity is not changed.
    void clear() {
        cuda_clear(_elements, _size);
        _size = 0;
    }

    // Inserts element at the back
    void push_back(const T& val) {
        if (_capacity == 0) {
            reserve(8);
        }
        else if (_size == _capacity) {
            reserve(2 * _capacity);
        }

        _elements[_size] = val;
        ++_size;
    }

    // Removes the last element from the Vector
    void pop_back() {
        if (_size == 0) return;
        --_size;
        _elements[_size].~T();
    }
    /*----------------------------*/

    /*--------Element Access--------*/

    // Access elements with bounds checking
    T& at(int n) {
        if (n < 0 || n >= _size) throw std::out_of_range("Index out of range.");
        return _elements[n];
    }

    // Access elements with bounds checking for constant Vectors.
    const T& at(int n) const {
        if (n < 0 || n >= _size) throw std::out_of_range("Index out of range.");
        return _elements[n];
    }

    // Access elements, no bounds checking
    T& operator[](size_t i) {
        return _elements[i];
    }

    // Returns a reference to the first element
    T& front() {
        return _elements[0];
    }

    // Returns a reference to the first element
    const T& front() const {
        return _elements[0];
    }

    // Returns a reference to the last element
    T& back() {
        return _elements[_size - 1];
    }

    // Returns a reference to the last element
    const T& back() const {
        return _elements[_size - 1];
    }

    // Returns a pointer to the array used by Vector
    T* data() {
        return _elements;
    }

    // Returns a pointer to the array used by Vector
    const T* data() const {
        return _elements;
    }
    /*----------------------------*/

    /*--------Arithmetic Operations--------*/
    template <typename Expr>
    Vector<T>& operator+=(const VectorExpression<T>& expr) {
        return apply_expr_op(expr, [](T& a, T b) { a += b; });
    }

    template <typename Expr>
    Vector<T>& operator-=(const VectorExpression<T>& expr) {
        return apply_expr_op(expr, [](T& a, T b) { a -= b; });
    }

    template <typename Expr>
    Vector<T>& operator*=(const VectorExpression<T>& expr) {
        return apply_expr_op(expr, [](T& a, T b) { a *= b; });
    }

    T dot(const Vector<T>& other) const {
        if (_size != other._size) throw std::invalid_argument("Size mismatch");
        T* temp = (int*)malloc(sizeof * temp);
        *temp = 0;
        cuda_dot(_elements, other._elements, _size, temp);
        return *temp;
    }





    Vector<T>& operator+=(const T& scalar) {
        cuda_scalar_op(_elements, scalar, _size, Add<T>());
        return *this;
    }

    Vector<T>& operator-=(const T& scalar) {
        cuda_scalar_op(_elements, scalar, _size, Subtract<T>());
        return *this;
    }

    Vector<T>& operator*=(const T& scalar) {
        cuda_scalar_op(_elements, scalar, _size, Multiply<T>());
        return *this;
    }

    Vector<T>& operator/=(const T& scalar) {
        cuda_scalar_op(_elements, scalar, _size, Divide<T>());
        return *this;
    }
    /*----------------------------*/

    /*--------Iterator--------*/
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        Iterator(T* p) : _curr(p) {}

        T& operator*() {
            return *(_curr);
        }

        Iterator& operator++() {
            _curr++;
            return *this;
        }

        Iterator& operator--() {
            _curr--;
            return *this;
        }

        Iterator operator+(int n) const {
            return Iterator(_curr + n);
        }

        Iterator& operator+=(int n) {
            _curr += n;
            return *this;
        }

        difference_type operator-(const Iterator& other) const {
            return _curr - other._curr;
        }

        bool operator==(const Iterator& other) const {
            return _curr == other._curr;
        }

        bool operator!=(const Iterator& other) const {
            return _curr != other._curr;
        }

    private:
        T* _curr;
    };

    Iterator begin() {
        return Iterator(&(_elements[0]));
    }

    Iterator end() {
        return Iterator(&(_elements[_size]));
    }

    const Iterator begin() const {
        return Iterator(&(_elements[0]));
    }

    const Iterator end() const {
        return Iterator(&(_elements[_size]));
    }

    //is there a need for cbegin() and cend():: future considerations
    /*----------------------------*/

private:
    size_t _size;
    T* _elements;
    size_t _capacity;

    //helper functions for arithmetic operations
    template<typename Op>
    Vector<T>& apply_expr_op(const VectorExpression<T>& expr, Op op) {
        if (_size != expr.size()) throw std::invalid_argument("Size mismatch");

        for (size_t i = 0; i < _size; ++i) {
            op(_elements[i], expr[i]);
        }
        return *this;
    }

};

/*-----------Comparaison Operators-----------*/
template<typename T>
bool operator==(const Vector<T>& a, const Vector<T>& b) {
    if (a.size() != b.size()) return false;
    bool* equality = (bool*)malloc(sizeof * equality);
    *equality = true;
    cuda_compare_equality(a.data(), b.data(), a.size(), equality);
    return *equality;
}

template<typename T>
bool operator!=(const Vector<T>& a, const Vector<T>& b) {
    return !(a == b);
}
/*-------------------------------------------*/

/*--------Scalar Operations---------*/
template<typename T>
Vector<T> operator+(const Vector<T>& v, const T& scalar) {
    Vector<T> result(v);
    result += scalar;
    return result;
}

template<typename T>
Vector<T> operator+(const T& scalar, const Vector<T>& v) {
    return v + scalar;
}

template<typename T>
Vector<T> operator-(const Vector<T>& v, const T& scalar) {
    Vector<T> result(v);
    result -= scalar;
    return result;
}

template<typename T>
Vector<T> operator-(const T& scalar, const Vector<T>& v) {
    Vector<T> result(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        result[i] = scalar - v[i];
    return result;
}

template<typename T>
Vector<T> operator*(const Vector<T>& v, const T& scalar) {
    Vector<T> result(v);
    result *= scalar;
    return result;
}

template<typename T>
Vector<T> operator*(const T& scalar, const Vector<T>& v) {
    return v * scalar;
}


template <typename T>
Vector<T> operator+(const Vector<T>& a, const Vector<T>& b) {
    assert(a.size() == b.size());

    Vector<T> res(a);

    cuda_vector_op(res.data(), b.data(), a.size(), Add<T>());

    return res;
}

template <typename T>
Vector<T> operator+=(Vector<T>& a, const Vector<T>& b) {
    assert(a.size() == b.size());

    cuda_vector_op(a.data(), b.data(), a.size(), Add<T>());

    return res;
}

template <typename T>
Vector<T> operator-(const Vector<T>& a, const Vector<T>& b) {
    assert(a.size() == b.size());

    Vector<T> res(a);

    cuda_vector_op(res.data(), b.data(), a.size(), Subtract<T>());

    return res;
}

template <typename T>
Vector<T> operator-=(const Vector<T>& a, const Vector<T>& b) {
    assert(a.size() == b.size());

    cuda_vector_op(a.data(), b.data(), a.size(), Subtract<T>());

    return res;
}

template <typename T>
Vector<T> operator*(const Vector<T>& a, const Vector<T>& b) {
    assert(a.size() == b.size());

    Vector<T> res(a);

    cuda_vector_op(res.data(), b.data(), a.size(), Multiply<T>());

    return res;
}

template <typename T>
Vector<T> operator*=(const Vector<T>& a, const Vector<T>& b) {
    assert(a.size() == b.size());


    cuda_vector_op(a.data(), b.data(), a.size(), Multiply<T>());

    return res;
}


template<typename T>
Vector<T> operator/(const Vector<T>& v, const T& scalar) {
    Vector<T> result(v);
    result /= scalar;
    return result;
}

#endif //VECTORCUDA_CUH
