#ifndef VECTOR_H
#define VECTOR_H


#include <cstddef>
#include <immintrin.h>
#include "VectorExpression.h"

template<typename T>
class Vector : public VectorExpression<T> {
public:
    /*--------Constructors--------*/

    // Default Constructor
    Vector(): _size(0), _elements(nullptr), _capacity(0) {
    }

    // //Default initialisation of values--more safe--for large vector it will be slow
    //  Vector(int size): _size(size), _elements(new T[size]), _capacity(size){
    //     for(int i =0 ; i<_size ; i++){
    //         _elements[i] = T();
    //     }
    // }

    explicit Vector(int capacity) : _size(0), _elements(new T[capacity]), _capacity(capacity) {
    }

    // Copy Constructor
    Vector(const Vector<T> &other) : _size(other._size), _elements(new T[other._capacity]), _capacity(other._capacity) {
        std::copy(other.begin(), other.end(), _elements);
    }

    // Move Constructor
    Vector(Vector<T> &&other) noexcept : _size(other._size), _elements(other._elements), _capacity(other._capacity){
        other._size = 0;
        other._elements = nullptr;
        other._capacity = 0;
    }

    //Move Assignment Operator
    Vector<T> &operator=(Vector<T> &&other) noexcept {
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
    Vector<T> &operator=(const Vector<T> &other) {
        if (this != &other) {
            delete[] _elements;

            _size = other._size;
            _capacity = other._capacity;
            _elements = new T[_capacity];

            std::copy(other.begin(), other.end(), _elements);
        }
        return *this;
    }

    // Destructor
    ~Vector() override {
        delete[] _elements;
        _elements = nullptr;
        _size = 0;
        _capacity = 0;
    }

    /*----------------------------*/

    /*------Expression Vector-----*/

    //Expression Based Constructor
    Vector(const VectorExpression<T> &expr) {
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

    [[nodiscard]] size_t size() const override {
        return _size;
    }

    /*-----------------------------*/

    /*--------Capacity--------*/

    [[nodiscard]] size_t capacity() const {
        return _capacity;
    }

    [[nodiscard]] bool empty() const {
        return _size == 0;
    }

    [[nodiscard]] size_t max_size() const {
        return _capacity;
    }

    // Requests a change in capacity
    // reserve() will never decrease the capacity.
    void reserve(int new_alloc) {
        if (new_alloc <= 0) return;
        if (new_alloc <= _capacity) return;

        T *tmp = new T[new_alloc];

        std::move(begin(), end(), tmp);

        delete[] _elements;
        _elements = tmp;
        _capacity = new_alloc;
    }

    //default initialisation and destroying the values can affect the performance to be considered
    // Changes the Vector's size.
    // If the new size is smaller, the last elements will be destroyed.
    // Has a default value param for custom values when resizing.
    void resize(int new_size, T val = T()) {
        if (new_size < 0) return;
        if (new_size > _capacity) {
            reserve(new_size);
        }

        if (new_size > _size) {
            for (int i = _size; i < new_size; i++) {
                _elements[i] = val;
            }
        } else if (new_size < _size) {
            for (size_t i = new_size; i < _size; i++) {
                _elements[i].~T();
            }
        }

        _size = new_size;
    }

    /*----------------------------*/

    /*----------Modifiers---------*/

    // Removes all elements from the Vector
    // Capacity is not changed.
    void clear() {
        for (int i = 0; i < _size; i++) {
            _elements[i].~T();
        }
        _size = 0;
    }

    // Inserts element at the back
    void push_back(const T &val) {
        if (_capacity == 0) {
            reserve(8);
        } else if (_size == _capacity) {
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
    T &at(int n) {
        if (n < 0 || n >= _size) throw std::out_of_range("Index out of range.");
        return _elements[n];
    }

    // Access elements with bounds checking for constant Vectors.
    const T &at(int n) const {
        if (n < 0 || n >= _size) throw std::out_of_range("Index out of range.");
        return _elements[n];
    }

    // Access elements, no bounds checking
    T &operator[](size_t i) {
        return _elements[i];
    }

    // Returns a reference to the first element
    T &front() {
        return _elements[0];
    }

    // Returns a reference to the first element
    const T &front() const {
        return _elements[0];
    }

    // Returns a reference to the last element
    T &back() {
        return _elements[_size - 1];
    }

    // Returns a reference to the last element
    const T &back() const {
        return _elements[_size - 1];
    }

    // Returns a pointer to the array used by Vector
    T *data() {
        return _elements;
    }

    // Returns a pointer to the array used by Vector
    const T *data() const {
        return _elements;
    }

    /*----------------------------*/

    /*--------Arithmetic Operations--------*/
    template<typename Expr>
    Vector<T> &operator+=(const VectorExpression<T> &expr) {
        return apply_expr_op(expr, [](T &a, T b) { a += b; });
    }

    template<typename Expr>
    Vector<T> &operator-=(const VectorExpression<T> &expr) {
        return apply_expr_op(expr, [](T &a, T b) { a -= b; });
    }

    template<typename Expr>
    Vector<T> &operator*=(const VectorExpression<T> &expr) {
        return apply_expr_op(expr, [](T &a, T b) { a *= b; });
    }

    T dot(const Vector<T> &other) const {
        if (_size != other._size) {
            throw std::invalid_argument("Size mismatch");
        }

        if constexpr (std::is_same_v<T, float>) {
            float result = 0.0f;

#pragma omp parallel
            {
                __m256 local_sum = _mm256_setzero_ps();

#pragma omp for
                for (int i = 0; i <= _size - 8; i += 8) {
                    __m256 a = _mm256_loadu_ps(&_elements[i]);
                    __m256 b = _mm256_loadu_ps(&other._elements[i]);
                    local_sum = _mm256_fmadd_ps(a, b, local_sum);
                }

                float temp[8];
                _mm256_storeu_ps(temp, local_sum);
                float thread_sum = temp[0] + temp[1] + temp[2] + temp[3] +
                                   temp[4] + temp[5] + temp[6] + temp[7];


#pragma omp for reduction(+:thread_sum)
                for (int i = (_size / 8) * 8; i < _size; ++i) {
                    thread_sum += _elements[i] * other._elements[i];
                }

#pragma omp atomic
                result += thread_sum;
            }

            return result;
        } else if constexpr (std::is_same_v<T, double>) {
            double result = 0.0;

#pragma omp parallel
            {
                __m256d local_sum = _mm256_setzero_pd();

#pragma omp for
                for (int i = 0; i <= _size - 4; i += 4) {
                    __m256d a = _mm256_loadu_pd(&_elements[i]);
                    __m256d b = _mm256_loadu_pd(&other._elements[i]);
                    local_sum = _mm256_fmadd_pd(a, b, local_sum);
                }

                double temp[4];
                _mm256_storeu_pd(temp, local_sum);
                double thread_sum = temp[0] + temp[1] + temp[2] + temp[3];


#pragma omp for reduction(+:thread_sum)
                for (int i = (_size / 4) * 4; i < _size; ++i) {
                    thread_sum += _elements[i] * other._elements[i];
                }

#pragma omp atomic
                result += thread_sum;
            }

            return result;
        } else {
            T result = T();

#pragma omp parallel for reduction(+:result)
            for (int i = 0; i < _size; ++i) {
                result += _elements[i] * other._elements[i];
            }

            return result;
        }
    }


    Vector &operator+=(const T &scalar) {
        if constexpr (std::is_same_v<T, float>) {
            __m256 scalar_vec = _mm256_set1_ps(scalar);

#pragma omp parallel for
            for (size_t i = 0; i + 8 <= _size; i += 8) {
                __m256 vec = _mm256_loadu_ps(&_elements[i]);
                vec = _mm256_add_ps(vec, scalar_vec);
                _mm256_storeu_ps(&_elements[i], vec);
            }


#pragma omp parallel for
            for (size_t i = (_size / 8) * 8; i < _size; ++i) {
                _elements[i] += scalar;
            }
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d scalar_vec = _mm256_set1_pd(scalar);

#pragma omp parallel for
            for (size_t i = 0; i + 4 <= _size; i += 4) {
                __m256d vec = _mm256_loadu_pd(&_elements[i]);
                vec = _mm256_add_pd(vec, scalar_vec);
                _mm256_storeu_pd(&_elements[i], vec);
            }


#pragma omp parallel for
            for (size_t i = (_size / 4) * 4; i < _size; ++i) {
                _elements[i] += scalar;
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < _size; ++i) {
                _elements[i] += scalar;
            }
        }

        return *this;
    }


    Vector &operator-=(const T &scalar) {
        if constexpr (std::is_same_v<T, float>) {
            __m256 scalar_vec = _mm256_set1_ps(scalar);

#pragma omp parallel for
            for (size_t i = 0; i + 8 <= _size; i += 8) {
                __m256 vec = _mm256_loadu_ps(&_elements[i]);
                vec = _mm256_sub_ps(vec, scalar_vec);
                _mm256_storeu_ps(&_elements[i], vec);
            }

#pragma omp parallel for
            for (size_t i = (_size / 8) * 8; i < _size; ++i) {
                _elements[i] -= scalar;
            }
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d scalar_vec = _mm256_set1_pd(scalar);

#pragma omp parallel for
            for (size_t i = 0; i + 4 <= _size; i += 4) {
                __m256d vec = _mm256_loadu_pd(&_elements[i]);
                vec = _mm256_sub_pd(vec, scalar_vec);
                _mm256_storeu_pd(&_elements[i], vec);
            }

#pragma omp parallel for
            for (size_t i = (_size / 4) * 4; i < _size; ++i) {
                _elements[i] -= scalar;
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < _size; ++i) {
                _elements[i] -= scalar;
            }
        }

        return *this;
    }


    Vector<T> &operator*=(const T &scalar) {
        if constexpr (std::is_same_v<T, float>) {
            __m256 scalar_vec = _mm256_set1_ps(scalar);

#pragma omp parallel for
            for (size_t i = 0; i + 8 <= _size; i += 8) {
                __m256 vec = _mm256_loadu_ps(&_elements[i]);
                vec = _mm256_mul_ps(vec, scalar_vec);
                _mm256_storeu_ps(&_elements[i], vec);
            }

#pragma omp parallel for
            for (size_t i = (_size / 8) * 8; i < _size; ++i) {
                _elements[i] *= scalar;
            }
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d scalar_vec = _mm256_set1_pd(scalar);

#pragma omp parallel for
            for (size_t i = 0; i + 4 <= _size; i += 4) {
                __m256d vec = _mm256_loadu_pd(&_elements[i]);
                vec = _mm256_mul_pd(vec, scalar_vec);
                _mm256_storeu_pd(&_elements[i], vec);
            }

#pragma omp parallel for
            for (size_t i = (_size / 4) * 4; i < _size; ++i) {
                _elements[i] *= scalar;
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < _size; ++i) {
                _elements[i] *= scalar;
            }
        }

        return *this;
    }

    Vector<T> &operator/=(const T &scalar) {
        if constexpr (std::is_same_v<T, float>) {
            __m256 scalar_vec = _mm256_set1_ps(scalar);

#pragma omp parallel for
            for (size_t i = 0; i + 8 <= _size; i += 8) {
                __m256 vec = _mm256_loadu_ps(&_elements[i]);
                vec = _mm256_div_ps(vec, scalar_vec);
                _mm256_storeu_ps(&_elements[i], vec);
            }

#pragma omp parallel for
            for (size_t i = (_size / 8) * 8; i < _size; ++i) {
                _elements[i] /= scalar;
            }
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d scalar_vec = _mm256_set1_pd(scalar);

#pragma omp parallel for
            for (size_t i = 0; i + 4 <= _size; i += 4) {
                __m256d vec = _mm256_loadu_pd(&_elements[i]);
                vec = _mm256_div_pd(vec, scalar_vec);
                _mm256_storeu_pd(&_elements[i], vec);
            }

#pragma omp parallel for
            for (size_t i = (_size / 4) * 4; i < _size; ++i) {
                _elements[i] /= scalar;
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < _size; ++i) {
                _elements[i] /= scalar;
            }
        }

        return *this;
    }


    /*----------------------------*/

    /*--------Iterator--------*/
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T *;
        using reference = T &;

        Iterator(T *p) : _curr(p) {
        }

        T &operator*() {
            return *(_curr);
        }

        Iterator &operator++() {
            ++_curr;
            return *this;
        }

        Iterator &operator--() {
            --_curr;
            return *this;
        }

        Iterator operator+(int n) const {
            return Iterator(_curr + n);
        }

        Iterator &operator+=(int n) {
            _curr += n;
            return *this;
        }

        difference_type operator-(const Iterator &other) const {
            return _curr - other._curr;
        }

        bool operator==(const Iterator &other) const {
            return _curr == other._curr;
        }

        bool operator!=(const Iterator &other) const {
            return _curr != other._curr;
        }

    private:
        T *_curr;
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

private:
    size_t _size;
    T *_elements;
    size_t _capacity;


    template<typename Op>
    Vector<T> &apply_expr_op(const VectorExpression<T> &expr, Op op) {
        if (_size != expr.size()) {
            throw std::invalid_argument("Size mismatch");
        }

        if constexpr (std::is_same_v<T, float>) {
#pragma omp parallel for
            for (size_t i = 0; i + 8 <= _size; i += 8) {
                __m256 vec_a = _mm256_loadu_ps(&_elements[i]);
                __m256 vec_b = _mm256_loadu_ps(&expr[i]);

                op(vec_a, vec_b);
                _mm256_storeu_ps(&_elements[i], vec_a);
            }


#pragma omp parallel for
            for (size_t i = (_size / 8) * 8; i < _size; ++i) {
                op(_elements[i], expr[i]);
            }
        } else if constexpr (std::is_same_v<T, double>) {
#pragma omp parallel for
            for (size_t i = 0; i + 4 <= _size; i += 4) {
                __m256d vec_a = _mm256_loadu_pd(&_elements[i]);
                __m256d vec_b = _mm256_loadu_pd(&expr[i]);

                op(vec_a, vec_b);
                _mm256_storeu_pd(&_elements[i], vec_a);
            }


#pragma omp parallel for
            for (size_t i = (_size / 4) * 4; i < _size; ++i) {
                op(_elements[i], expr[i]);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < _size; ++i) {
                op(_elements[i], expr[i]);
            }
        }

        return *this;
    }
};

template<typename T>
bool operator==(const Vector<T> &a, const Vector<T> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

template<typename T>
bool operator!=(const Vector<T> &a, const Vector<T> &b) {
    return !(a == b);
}

/*-------------------------------------------*/

/*--------Scalar Operations---------*/
template<typename T>
Vector<T> operator+(const Vector<T> &v, const T &scalar) {
    Vector<T> result(v);
    result += scalar;
    return result;
}

template<typename T>
Vector<T> operator+(const T &scalar, const Vector<T> &v) {
    return v + scalar;
}

template<typename T>
Vector<T> operator-(const Vector<T> &v, const T &scalar) {
    Vector<T> result(v);
    result -= scalar;
    return result;
}

template<typename T>
Vector<T> operator-(const T &scalar, const Vector<T> &v) {
    Vector<T> result(v.size());

    if constexpr (std::is_same_v<T, float>) {
        __m256 scalar_vec = _mm256_set1_ps(scalar);

#pragma omp parallel for
        for (size_t i = 0; i + 8 <= v.size(); i += 8) {
            __m256 vec = _mm256_loadu_ps(&v[i]);
            vec = _mm256_sub_ps(scalar_vec, vec);
            _mm256_storeu_ps(&result[i], vec);
        }


#pragma omp parallel for
        for (size_t i = (v.size() / 8) * 8; i < v.size(); ++i) {
            result[i] = scalar - v[i];
        }
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d scalar_vec = _mm256_set1_pd(scalar);

#pragma omp parallel for
        for (size_t i = 0; i + 4 <= v.size(); i += 4) {
            __m256d vec = _mm256_loadu_pd(&v[i]);
            vec = _mm256_sub_pd(scalar_vec, vec);
            _mm256_storeu_pd(&result[i], vec);
        }


#pragma omp parallel for
        for (size_t i = (v.size() / 4) * 4; i < v.size(); ++i) {
            result[i] = scalar - v[i];
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = scalar - v[i];
        }
    }

    return result;
}


template<typename T>
Vector<T> operator*(const Vector<T> &v, const T &scalar) {
    Vector<T> result(v);
    result *= scalar;
    return result;
}

template<typename T>
Vector<T> operator*(const T &scalar, const Vector<T> &v) {
    return v * scalar;
}

template<typename T>
Vector<T> operator/(const Vector<T> &v, const T &scalar) {
    Vector<T> result(v);
    result /= scalar;
    return result;
}

#endif
