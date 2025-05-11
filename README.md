# Matrix Module Documentation

This document provides comprehensive documentation for using the `matrix_cpu` Python module, which provides efficient matrix operations implemented in C++ and exposed to Python via pybind11.

## Table of Contents

- [Installation](#installation)
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
- `set(row, col, value)`: Set the element at the specified row and column.
- `setZero()`: Fill the matrix with zeros.
- `setOnes()`: Fill the matrix with ones.
- `setConstant(value)`: Fill the matrix with a constant value.
- `sum()`: Compute the sum of all elements.
- `transpose()`: Return a transposed view of the matrix.
- `dot(other)`: Compute the dot product with another matrix.
- `normalized()`: Return a normalized version of the matrix.

#### Operators

- `matrix(row, col)`: Access the element at the specified row and column.
- `matrix + matrix`: Add two matrices (returns a lazy expression).
- `matrix - matrix`: Subtract two matrices (returns a lazy expression).
- `matrix * matrix`: Multiply two matrices (returns a lazy expression).

### Matrix Expressions

Matrix operations return lazy expressions that can be evaluated when needed. The following expression types are supported:

- `MatrixAdd<T>`: Result of matrix addition.
- `MatrixSub<T>`: Result of matrix subtraction.
- `MatrixMul<T>`: Result of matrix multiplication.
- `MatrixTranspose<T>`: Result of matrix transposition.

#### Methods for Matrix Expressions

- `rows()`: Get the number of rows.
- `columns()`: Get the number of columns.
- `eval()`: Evaluate the expression and return a concrete Matrix.
- `expr(row, col)`: Access the element at the specified row and column.

### `MatrixTranspose<T>` Class

The `MatrixTranspose<T>` class is available as `MatrixTransposeInt`, `MatrixTransposeFloat`, and `MatrixTransposeDouble` in Python.

#### Methods

- `rows()`: Get the number of rows.
- `columns()`: Get the number of columns.
- `eval()`: Evaluate the transpose expression and return a concrete Matrix.

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

### Example 3: Matrix Operators (Addition, Subtraction, Multiplication)

```python
import matrix_cpu as mc

# Create matrices
a = mc.MatrixFloat(2, 2)
b = mc.MatrixFloat(2, 2)

# Set values for matrix a
a.set(0, 0, 1.0)
a.set(0, 1, 2.0)
a.set(1, 0, 3.0)
a.set(1, 1, 4.0)

# Set values for matrix b
b.set(0, 0, 5.0)
b.set(0, 1, 6.0)
b.set(1, 0, 7.0)
b.set(1, 1, 8.0)

print("Matrix a:")
print(a)
print("Matrix b:")
print(b)

# Addition
c_add = a + b
print("\nAddition (a + b):")
print(c_add)  # This is a lazy expression
print("Evaluated result:")
print(c_add.eval())  # Evaluate to get a concrete matrix

# Subtraction
c_sub = a - b
print("\nSubtraction (a - b):")
print(c_sub)  # This is a lazy expression
print("Evaluated result:")
print(c_sub.eval())  # Evaluate to get a concrete matrix

# Matrix multiplication
c_mul = a * b
print("\nMatrix multiplication (a * b):")
print(c_mul)  # This is a lazy expression
print("Evaluated result:")
print(c_mul.eval())  # Evaluate to get a concrete matrix
```

### Example 4: Chaining Operations

```python
import matrix_cpu as mc

# Create matrices
a = mc.MatrixFloat(2, 2)
b = mc.MatrixFloat(2, 2)
c = mc.MatrixFloat(2, 2)

# Set values
a.set(0, 0, 1.0)
a.set(0, 1, 2.0)
a.set(1, 0, 3.0)
a.set(1, 1, 4.0)

b.set(0, 0, 5.0)
b.set(0, 1, 6.0)
b.set(1, 0, 7.0)
b.set(1, 1, 8.0)

c.set(0, 0, 9.0)
c.set(0, 1, 10.0)
c.set(1, 0, 11.0)
c.set(1, 1, 12.0)

# Chain operations: (a + b) * c
# Note that the operations are lazy, so the intermediate result (a + b) isn't computed
# until the entire expression is evaluated
result = (a + b) * c

print("Matrix a:")
print(a)
print("Matrix b:")
print(b)
print("Matrix c:")
print(c)
print("\nResult of (a + b) * c:")
print(result.eval())
```
