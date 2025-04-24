# Matrix and Vector Library Documentation

This library implements a CPU-based matrix and vector system in modern C++ using **expression templates** for lazy evaluation and a flat memory layout for performance. Both components support composable arithmetic operations and efficient iteration.

## Shared Architecture

- **Expression Templates**: Enable lazy evaluation of operations like `A + B`, `v1 * v2`, or `A.transpose() * B`, minimizing temporary allocations.
- **Flat Memory Layout**: Data is stored in 1D arrays for cache-friendly access and SIMD compatibility.
- **STL-Style Iterators**: Random access iterators are available for both matrices and vectors.

## Matrix Class (`Matrix<T>`)

- Dynamic-size 2D container
- Arithmetic with scalars and other matrices
- Transpose support via `MatrixTranspose`
- Dot product, normalization, sum, min/max coefficients
- Non-owning views:
  - **Row**: `A.row(i)`
  - **Column**: `A.col(j)`
  - **Block**: `A.block(i, j, r, c)`
- Dynamic resizing while preserving values

## Vector Class (`Vector<T>`)

- Dynamically resizable 1D container
- Arithmetic with scalars and other vectors
- Dot product and scalar ops via `VectorExpression<T>`
- Push-back and clear support for host-side manipulation

## Example Usage

```cpp
Matrix<float> A(2, 2), B(2, 2);
A.setConstant(1.0f);
B.setConstant(2.0f);

Matrix<float> C = A + B;
std::cout << C;

Vector<float> v1 = {1, 2, 3};
Vector<float> v2 = v1 * 2.0f;
float dot = v1.dot(v2);
