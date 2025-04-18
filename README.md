# CUDA Vector Library Documentation

This part of the project implements a CUDA-accelerated vector class in C++ that supports expression templates and asynchronous GPU operations for mathematical operations such as addition, scalar operations, dot product, and more.

---

## Features

- **Expression Templates**: Lazy evaluation of arithmetic operations like `v + w`, `v * 2`, etc.
- **CUDA Kernels**: Parallel implementations for:
  - `copyKernel`: Deep copy from one device array to another
  - `scalarOpKernel`: Applies scalar operations (add, subtract, multiply, divide)
  - `compareEqKernel`: Checks equality of two vectors
  - `dotKernel`: Calculates dot product using shared memory and reduction
  - `clearKernel`: Resets vector to default value
- **Grid-Stride Looping**: Optimized thread distribution across blocks and grids
- **CUDA Streams**: All kernel wrappers use asynchronous execution via `cudaStream_t`
- **Shared Memory Reduction**: Efficient block-wise sum for dot product

---

## Core Components

### `Vector<T>`
A dynamically resizable array supporting:
- Host-side manipulation (e.g., push_back, clear)
- Arithmetic with scalar or another `Vector<T>`
- Expression-based construction

### `VectorExpression<T>`
Abstract base class used for expression templates like `v + w`. It's **host-only** (can’t be run in device code), so evaluation must be done before passing to GPU.

---

## Kernel Wrappers

Each function below launches a CUDA kernel with proper grid/block setup and uses **asynchronous streams** for performance.

### `cuda_copy<T>(from, to, size)`
Copies a vector from host to device and back.

### `cuda_scalar_op<T, Op>(elts, scalar, size, op)`
Applies a scalar operation to each element using a functor (e.g., `Add`, `Multiply`).

### `cuda_clear<T>(elts, size)`
Resets all elements of the array to default (`T()`).

### `cuda_dot<T>(elts1, elts2, size, result)`
Computes the dot product of two vectors using shared memory reduction.

### `cuda_compare_equality<T>(elts1, elts2, size, equality)`
Checks whether two vectors are element-wise equal.

---

## Kernel Design Patterns

Each kernel follows this thread-safe indexing pattern:

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = index; i < size; i += stride) {
    // thread work
}
```

 Efficient and scalable for any array size.

---

## Functor Structs

```cpp
struct Add      { __device__ void operator()(int& a, int b) const { a += b; } };
struct Subtract { __device__ void operator()(int& a, int b) const { a -= b; } };
struct Multiply { __device__ void operator()(int& a, int b) const { a *= b; } };
struct Divide   { __device__ void operator()(int& a, int b) const { a /= b; } };
```

These are used in scalar operations via template dispatch.

---

## Example Usage (Main)

```cpp
Vector<int> v;
v.push_back(10);
v.push_back(20);

Vector<int> v1;
v1.push_back(1);
v1.push_back(2);

auto result = v + v1;  // Uses expression template
int dot = v.dot(v1);   // Uses CUDA kernel
```

---

## Build and Run

Compile with:
```bash
nvcc -std=c++17 Vector.cu -o vector_cuda
```

---



