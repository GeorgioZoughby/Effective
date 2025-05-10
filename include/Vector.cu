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


template <typename T>
__global__ void copyKernel(const T* from, T* to, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        to[i] = from[i];
    }
}


template <typename Op, typename T>
__global__ void scalarOpKernel(T* elts, const T scal, int size, Op op) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        op(elts[i], scal);
    }
}


template <typename T>
__global__ void compareEqKernel(const T* elts1, const T* elts2, bool* equality, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        if (elts1[i] != elts2[i]) {
            *equality = false;
        }
    }
}


template <typename T>
__global__ void dotKernel(const T* elts1, const T* elts2, T* result, unsigned int size)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    T sum = 0;

    // Each thread accumulates its own partial sum
    for (int i = index; i < size; i += stride) {
        sum += elts1[i] * elts2[i];
    }

    // Store partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's result atomically to global result
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}


template <typename T>
__global__ void clearKernel(T* elts, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        elts[i] = T();  // reset to default
    }
}

template <typename Op, typename T>
__global__ void vectorOpKernel(T* elts1, const T* elts2, unsigned int size, Op op) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        op(elts1[index], elts2[index]);
    }
}



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


using namespace expr;
#include <chrono>

//int main() {
//    Vector<int> v;
//
//    // Start measuring time
//    auto start = std::chrono::high_resolution_clock::now();
//
//    std::cout << "Pushing 3 elements...\n";
//    v.push_back(10);
//    v.push_back(20);
//    v.push_back(30);
//
//    std::cout << "Front: " << v.front() << "\n";
//    std::cout << "Back: " << v.back() << "\n";
//    std::cout << "At(1): " << v.at(1) << "\n";
//
//    std::cout << "Using operator[]: ";
//    for (int i = 0; i < v.size(); ++i) std::cout << v[i] << " ";
//    std::cout << "\n";
//
//    std::cout << "Size: " << v.size() << ", Capacity: " << v.capacity() << "\n";
//
//    std::cout << "Pop back...\n";
//    v.pop_back();
//    std::cout << "Size: " << v.size() << ", Back: " << v.back() << "\n";
//
//    std::cout << "Resizing to 5 with default value 99...\n";
//    v.resize(5, 99);
//    for (int i = 0; i < v.size(); ++i) std::cout << v[i] << " ";
//    std::cout << "\n";
//
//    std::cout << "Clearing vector...\n";
//    v.clear();
//    std::cout << "Size after clear: " << v.size() << ", Empty? " << v.empty() << "\n";
//
//    std::cout << "Testing iterators (push then iterate):\n";
//    for (int i = 1; i <= 5; ++i) v.push_back(i * 10);
//    for (auto it = v.begin(); it != v.end(); ++it)
//        std::cout << *it << " ";
//    std::cout << "\n";
//
//    Vector<int> v1;
//    for (int i = 0; i < 5; ++i) v1.push_back(i + 1);
//
//    // Arithmetic expressions
//    Vector<int> sum = v + v1;
//    Vector<int> diff = v - v1;
//    Vector<int> prod = v * v1;
//    Vector<int> scaled = v * 2;
//    Vector<int> shifted = v + 5;
//
//    std::cout << "v + v1: ";
//    for (int i = 0; i < sum.size(); ++i) std::cout << sum[i] << " ";
//    std::cout << "\nv - v1: ";
//    for (int i = 0; i < diff.size(); ++i) std::cout << diff[i] << " ";
//    std::cout << "\nv * v1: ";
//    for (int i = 0; i < prod.size(); ++i) std::cout << prod[i] << " ";
//    std::cout << "\nv * 2: ";
//    for (int i = 0; i < scaled.size(); ++i) std::cout << scaled[i] << " ";
//    std::cout << "\nv + 5: ";
//    for (int i = 0; i < shifted.size(); ++i) std::cout << shifted[i] << " ";
//    std::cout << "\n";
//    std::cout << "v + 5: size = " << shifted.size() << "\n";
//
//    std::cout << "Dot product v . v1: " << v.dot(v1) << "\n";
//
//    std::cout << "Equality test: v == v1? " << (v == v1 ? "true" : "false") << "\n";
//    std::cout << "Inequality test: v != v1? " << (v != v1 ? "true" : "false") << "\n";
//
//    // End measuring time
//    auto end = std::chrono::high_resolution_clock::now();
//
//    // Calculate the duration
//    std::chrono::duration<double> duration = end - start;
//    std::cout << "Time taken: " << duration.count() << " seconds\n";
//
//    return 0;
//}
//Time measure before pinned memory transfer: 1.33s
//Time measure after pinned memory transfer: 0.27s



// =======================
// Kernel Wrapper Functions with Streams
// =======================

// Each wrapper now uses a cudaStream_t for asynchronous operations

// Helper macro to define stream and clean up
#define CREATE_CUDA_STREAM(stream) \
    cudaStream_t stream; \
    cudaStreamCreate(&stream);

#define DESTROY_CUDA_STREAM(stream) \
    cudaStreamSynchronize(stream); \
    cudaStreamDestroy(stream);



template <typename T>
cudaError_t cuda_copy(const T* from, T* to, unsigned int size) {
    T* dev_from = nullptr;
    T* dev_to = nullptr;
    T* pinned_from = nullptr;
    T* pinned_to = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned host memory
    cudaStatus = cudaHostAlloc((void**)&pinned_from, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaHostAlloc((void**)&pinned_to, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy data into pinned buffer
    memcpy(pinned_from, from, size * sizeof(T));

    // Allocate device memory
    cudaStatus = cudaMalloc((void**)&dev_from, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_to, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer from host to device
    cudaStatus = cudaMemcpyAsync(dev_from, pinned_from, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    // Kernel launch
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    copyKernel<<<gridSize, blockSize, 0, stream>>>(dev_from, dev_to, size);

    // Async transfer from device to pinned host
    cudaStatus = cudaMemcpyAsync(pinned_to, dev_to, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    // Synchronize stream and copy to user buffer
    cudaStreamSynchronize(stream);
    memcpy(to, pinned_to, size * sizeof(T));

Error:
    // Clean up
    DESTROY_CUDA_STREAM(stream);
    if (dev_from) cudaFree(dev_from);
    if (dev_to) cudaFree(dev_to);
    if (pinned_from) cudaFreeHost(pinned_from);
    if (pinned_to) cudaFreeHost(pinned_to);

    return cudaStatus;
}

template <typename Op, typename T>
cudaError_t cuda_vector_op(T* elts1, const T* elts2, unsigned int size, Op op) {
    T* dev_elts1 = nullptr;
    T* dev_elts2 = nullptr;
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned host memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts1, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaHostAlloc((void**)&pinned_elts2, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));

    // Allocate device memory
    cudaStatus = cudaMalloc((void**)&dev_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer from pinned memory to device
    cudaStatus = cudaMemcpyAsync(dev_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    vectorOpKernel<<<numBlocks, blockSize, 0, stream>>>(dev_elts1, dev_elts2, size, op);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(pinned_elts1, dev_elts1, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    // Synchronize and copy from pinned memory to output
    cudaStreamSynchronize(stream);
    memcpy(elts1, pinned_elts1, size * sizeof(T));

Error:
    // Clean up
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_elts1);
    cudaFree(dev_elts2);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    return cudaStatus;
}



template <typename Op, typename T>
cudaError_t cuda_scalar_op(T* elts, const T scal, unsigned int size, Op op) {
    T* dev_elts = 0;
    T* pinned_elts = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned host memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts, elts, size * sizeof(T));

    cudaStatus = cudaMalloc((void**)&dev_elts, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer from pinned memory to device
    cudaStatus = cudaMemcpyAsync(dev_elts, pinned_elts, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    scalarOpKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts, scal, size, op);

    // Async transfer from device to pinned memory
    cudaStatus = cudaMemcpyAsync(pinned_elts, dev_elts, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    Error:
        // Synchronize and copy from pinned memory to output
        cudaStreamSynchronize(stream);
    memcpy(elts, pinned_elts, size * sizeof(T));

    // Clean up
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_elts);
    cudaFreeHost(pinned_elts);
    return cudaStatus;
}




template <typename T>
cudaError_t cuda_compare_equality(const T* elts1, const T* elts2, unsigned int size, bool* equality) {
    T* dev_elts1 = 0;
    T* dev_elts2 = 0;
    bool* dev_equality = 0;
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts1, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaHostAlloc((void**)&pinned_elts2, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));

    cudaStatus = cudaMalloc((void**)&dev_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_equality, sizeof(bool));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer to device
    cudaStatus = cudaMemcpyAsync(dev_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemsetAsync(dev_equality, 1, sizeof(bool), stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    compareEqKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts1, dev_elts2, dev_equality, size);

    // Async transfer back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(equality, dev_equality, sizeof(bool), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

Error:
    // Synchronize and clean up
    cudaStreamSynchronize(stream);
    cudaFree(dev_elts1);
    cudaFree(dev_elts2);
    cudaFree(dev_equality);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    return cudaStatus;
}




template <typename T>
cudaError_t cuda_dot(const T* elts1, const T* elts2, unsigned int size, T* result) {
    T* dev_elts1 = 0;
    T* dev_elts2 = 0;
    T* dev_result = 0;
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts1, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaHostAlloc((void**)&pinned_elts2, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));

    cudaStatus = cudaMalloc((void**)&dev_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_result, sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer to device
    cudaStatus = cudaMemcpyAsync(dev_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemsetAsync(dev_result, 0, sizeof(T), stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(T);
    dotKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(dev_elts1, dev_elts2, dev_result, size);

    // Async transfer back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(result, dev_result, sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

Error:
    // Synchronize and clean up
    cudaStreamSynchronize(stream);
    cudaFree(dev_elts1);
    cudaFree(dev_elts2);
    cudaFree(dev_result);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    return cudaStatus;
}




template <typename T>
cudaError_t cuda_clear(T* elts, unsigned int size) {
    T* dev_elts = 0;
    T* pinned_elts = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts, elts, size * sizeof(T));

    cudaStatus = cudaMalloc((void**)&dev_elts, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer to device
    cudaStatus = cudaMemcpyAsync(dev_elts, pinned_elts, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    clearKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts, size);

    // Async transfer back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(pinned_elts, dev_elts, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    Error:
        // Synchronize and clean up
        cudaStreamSynchronize(stream);
        memcpy(elts, pinned_elts, size * sizeof(T));
        cudaFree(dev_elts);
        cudaFreeHost(pinned_elts);
        return cudaStatus;
}
