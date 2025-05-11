# Matrix Module Documentation

This document provides comprehensive documentation for using the `matrix_cpu` Python module, which provides efficient matrix operations implemented in C++ and exposed to Python via pybind11.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Matrix Creation](#matrix-creation)
- [Matrix Operations](#matrix-operations)
- [Performance Benchmarks](#performance-benchmarks)
- [Examples](#examples)
- [API Reference](#api-reference)

## Installation

To install and use this module, follow these steps:

1. Clone the repository:

   ```bash
   git clone git@github.com:GeorgioZoughby/Effective.git Project_Effective
   cd Project_Effective
   ```

2. Create and activate a Python virtual environment (recommended):

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. You can avoid installing pybind11 since it was added to the repo and automatically linked in Cmake:

4. Build the module:

   ```bash
   mkdir -p build
   cd build
   cmake ..
   make
   ```

5. The built module (`matrix_cpu.*.so`) will be available in the `build` directory.

## API Reference

### The module name is matrix_cpu

### `Matrix<T>` Class

The `Matrix<T>` class is available as `MatrixInt`, `MatrixFloat`, and `MatrixDouble` in Python.

#### Constructor

- `Matrix(rows, columns)`: Create a matrix with the specified dimensions.

#### Methods

- `rows()`: Get the number of rows.
- `columns()`: Get the number of columns.
- `setZero()`: Fill the matrix with zeros.
- `setOnes()`: Fill the matrix with ones.
- `setConstant(value)`: Fill the matrix with a constant value.
- `sum()`: Compute the sum of all elements.
- `transpose()`: Return a transposed view of the matrix.
- `dot(other)`: Compute the dot product with another matrix.
- `normalized()`: Return a normalized version of the matrix.

#### Operators

- `matrix(row, col)`: Access or modify the element at the specified row and column.

### `MatrixTranspose<T>` Class

The `MatrixTranspose<T>` class is available as `MatrixTransposeMatrixInt`, `MatrixTransposeMatrixFloat`, and `MatrixTransposeMatrixDouble` in Python.

#### Methods

- `rows()`: Get the number of rows.
- `columns()`: Get the number of columns.

#### Operators

- `transpose(row, col)`: Access the element at the specified row and column.

## Examples

### Example 1: Creating and Manipulating Matrices

```python
import matrix_cpu

# Create a 3x3 matrix of floats
m = matrix_cpu.MatrixFloat(3, 3)

# Fill with a constant value
m.setConstant(2.5)
print("Original Matrix:")
print(m)

# Compute the sum
print(f"Sum of elements: {m.sum()}")  # Should be 22.5 for a 3x3 matrix with 2.5 in each cell

# Transpose the matrix
mt = m.transpose()
print("Transposed Matrix:")
print(mt)
```

### Example 2: Matrix Dot Product

```python
import matrix_cpu

# Create two matrices
a = matrix_cpu.MatrixFloat(2, 3)
a.setConstant(2.0)

b = matrix_cpu.MatrixFloat(3, 2)
b.setConstant(3.0)

# Compute the dot product
c = a.dot(b)
print("Matrix A (2x3):")
print(a)
print("Matrix B (3x2):")
print(b)
print("Dot product A·B (2x2):")
print(c)
```
