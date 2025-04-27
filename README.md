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

# Matrix CUDA Library Documentation

This library provides a CUDA-accelerated implementation of a `Matrix` class in C++. It supports various matrix operations using CUDA kernels, including matrix addition, multiplication, transposition, and element-wise operations. It also includes helper functions for matrix manipulation and basic linear system solvers.

---

## Features

- **Matrix Expressions**: Support for matrix operations like addition, multiplication, and element-wise operations using expression templates.
- **CUDA Kernels**: Parallelized implementations for matrix operations:
  - `copyKernel`: Copies data from one device array to another.
  - `scalarOpKernel`: Applies scalar operations (add, subtract, multiply, divide) to a matrix.
  - `matrixOpKernel`: Applies element-wise operations (addition, subtraction, multiplication) to matrices.
  - `matrixMultiplyKernel`: Multiplies two matrices.
  - `transposeKernel`: Computes the transpose of a matrix.
  - `clearKernel`: Resets matrix elements to default values.
- **Matrix Views**: Various matrix views such as row, column, and block views.
- **CUDA Streams**: All operations are asynchronous via CUDA streams for efficient execution.
- **Matrix Solvers**: Basic solver for linear systems using Gaussian elimination.

---

## Core Components

### `Matrix<T>`
A matrix class that supports:
- Host-side manipulation (e.g., `push_back`, `resize`).
- Arithmetic operations with scalars or other matrices (e.g., element-wise addition, multiplication).
- Expression-based matrix construction.
- Transposition and element-wise operations.

### `MatrixExpression<T>`
An abstract base class used for expression templates like `A + B`. This is **host-only** and used for lazy evaluation, enabling efficient execution.

---

## CUDA Kernels

### `copyKernel<T>(from, to, size)`
Copies matrix data from the host to the device or vice versa.

### `scalarOpKernel<T, Op>(elts, scalar, size, op)`
Applies a scalar operation (e.g., `AddOp`, `MultiplyOp`) to each element of the matrix.

### `matrixOpKernel<T, Op>(elts1, elts2, size, op)`
Performs element-wise operations (addition, subtraction, multiplication) between two matrices using CUDA streams.

### `matrixMultiplyKernel<T>(A, B, C, rowsA, colsA, colsB)`
Multiplies two matrices `A` and `B` to compute the result matrix `C`.

### `transposeKernel<T>(input, output, rows, cols)`
Computes the transpose of a matrix.

### `clearKernel<T>(elts, size)`
Resets all elements in the matrix to their default value (`T()`).

---

## Matrix Expression Classes

### `MatrixTranspose<T>`
Represents the transpose of a matrix. The `operator()` returns the transposed element at `(col, row)`.

### `MatrixCwiseProduct<T>`
Represents the element-wise product of two matrices. It asserts that both matrices have the same dimensions.

### `MatrixCwiseQuotient<T>`
Represents the element-wise quotient of two matrices. Like `MatrixCwiseProduct`, it checks for dimension compatibility.

### `MatrixRowView<T>`
Represents a single row of a matrix as a view. It is initialized with the row index and can access the matrix element at `(0, col)`.

### `MatrixColView<T>`
Represents a single column of a matrix as a view. It is initialized with the column index and can access the matrix element at `(row, 0)`.

### `MatrixBlockView<T>`
Represents a submatrix (block) view of the matrix. It allows access to a rectangular section of the matrix starting from `(startRow, startCol)`.

---

## Matrix Operations

### Arithmetic Operations
- **Matrix-Scalar Operations**: `operator+=`, `operator-=`, `operator*=`, and `operator/=` support scalar matrix operations.
- **Matrix-Matrix Operations**: Element-wise addition, subtraction, multiplication, and division using the corresponding operators (`operator+`, `operator-`, `operator*`, `operator/`).
  
### Linear Algebra
- **Matrix Transposition**: `transpose()` computes the transpose of the matrix.
- **Dot Product**: `dot()` computes the dot product of two matrices.
- **Matrix Multiplication**: `operator*()` computes the matrix multiplication.

### Basic Matrix Manipulations
- **Resize**: `resize()` changes the size of the matrix while maintaining its elements.
- **Set Values**: `setZero()`, `setOnes()`, `setRandom()`, and `setConstant()` are used to set matrix elements to zero, ones, random values, or a constant, respectively.

### Norms and Coefficients
- **Norm**: `norm()` computes the Frobenius norm of the matrix.
- **Min/Max Coefficients**: `minCoeff()` and `maxCoeff()` return the smallest and largest elements of the matrix.

---

## CUDA Interface Functions

### `cuda_matrix_copy<T>(from, to, size)`
Asynchronously copies matrix data between the host and device.

### `cuda_matrix_scalar_op<T, Op>(elts, scalar, size, op)`
Applies a scalar operation to each matrix element asynchronously using CUDA.

### `cuda_matrix_op<T, Op>(elts1, elts2, size, op)`
Performs element-wise operations (addition, subtraction, multiplication) between two matrices using CUDA streams.

### `cuda_matrix_multiply<T>(A, B, C, rowsA, colsA, colsB)`
Asynchronously multiplies two matrices `A` and `B` to compute the result matrix `C`.

### `cuda_matrix_transpose<T>(input, output, rows, cols)`
Asynchronously computes the transpose of a matrix using CUDA.

---

## Example Usage (Main)

```cpp
Matrix<int> mat1(3, 3);
mat1.setRandom();

Matrix<int> mat2(3, 3);
mat2.setRandom();

Matrix<int> result = mat1 * mat2; // Matrix multiplication

Matrix<int> transposed = mat1.transpose(); // Matrix transposition

Matrix<int> row_view = mat1.row(1); // Row view of the matrix
```

---

## Build and Run

To compile the project:

```bash
nvcc -std=c++17 Matrix.cu -o matrix_cuda
```

---




