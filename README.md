# Matrix and Vector Module Documentation

This document provides comprehensive documentation for using the `matrix_cpu` and `vector_cpu` Python modules, which provide efficient matrix and vector operations implemented in C++ and exposed to Python via pybind11.

## Table of Contents

- [Installation](#installation)
- [Matrix API Reference](#matrix-api-reference)
- [Vector API Reference](#vector-api-reference)
- [Examples](#examples)
  - [Matrix Examples](#matrix-examples)
  - [Vector Examples](#vector-examples)

## Installation

To install and use these modules, follow these steps:

1. Clone the repository:

   ```bash
   git clone --recurse-submodules git@github.com:GeorgioZoughby/Effective.git Project_Effective
   cd Project_Effective
   ```

2. Create and activate a Python virtual environment (recommended):

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. You can avoid installing pybind11 since it was added to the repo and automatically linked in Cmake.

4. Build the modules:

   ## Linux
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make
   ```
   
   ## Linux GPU ON
   ```bash
   mkdir -p build
   cd build
   cmake .. -DBUILD_CUDA=ON
   make
   ```
   
   ## Windows GPU ON
   ```bash
    cmake -S . -B build -DBUILD_CUDA=ON
    cmake --build build
   ```

   ## Windows GPU OFF
   ```bash
    cmake -S . -B build 
    cmake --build build
   ```

5. The built modules (`matrix_cpu.*.so`(or `.pyd`) and `vector_cpu.*.so` or (`.pyd`)) will be available in the `build` directory.

## Matrix API Reference

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

## Vector API Reference

### The module name is vector_cpu

### `Vector<T>` Class

The `Vector<T>` class is available as `VectorInt`, `VectorFloat`, and `VectorDouble` in Python.

#### Constructors

- `Vector()`: Create an empty vector.
- `Vector(capacity)`: Create a vector with the specified initial capacity.
- `Vector(other)`: Copy constructor to create a vector from another vector.

#### Capacity Methods

- `size()`: Get the number of elements in the vector.
- `capacity()`: Get the current capacity of the vector.
- `empty()`: Check if the vector is empty. Returns `True` if the vector contains no elements.
- `reserve(new_capacity)`: Request a change in capacity. Reserves space for at least `new_capacity` elements.
- `resize(new_size, value=default)`: Change the vector's size. If `new_size` is greater than the current size, the new elements are initialized with `value`.

#### Element Access Methods

- `operator[](i)`: Access element at index `i` (no bounds checking).
- `at(n)`: Access element at index `n` with bounds checking. Throws an exception if out of range.
- `front()`: Return a reference to the first element.
- `back()`: Return a reference to the last element.
- `data()`: Return a pointer to the array used by the vector.

#### Modifiers

- `clear()`: Remove all elements from the vector (capacity is not changed).
- `push_back(value)`: Add an element to the end of the vector.
- `pop_back()`: Remove the last element from the vector.

#### Arithmetic Operations

- `dot(other)`: Compute the dot product with another vector.
- `operator+=(expr)`: Add a vector expression to this vector.
- `operator-=(expr)`: Subtract a vector expression from this vector.
- `operator*=(expr)`: Element-wise multiply this vector by a vector expression.
- `operator+=(scalar)`: Add a scalar to each element of the vector.
- `operator-=(scalar)`: Subtract a scalar from each element of the vector.
- `operator*=(scalar)`: Multiply each element of the vector by a scalar.
- `operator/=(scalar)`: Divide each element of the vector by a scalar.

#### Operators

- `vector + vector`: Add two vectors (returns a lazy expression).
- `vector - vector`: Subtract two vectors (returns a lazy expression).
- `vector * vector`: Element-wise multiply two vectors (returns a lazy expression).
- `vector + scalar`: Add a scalar to each element.
- `scalar + vector`: Add a scalar to each element.
- `vector - scalar`: Subtract a scalar from each element.
- `scalar - vector`: Subtract each element from a scalar.
- `vector * scalar`: Multiply each element by a scalar.
- `scalar * vector`: Multiply each element by a scalar.
- `vector / scalar`: Divide each element by a scalar.
- `vector == vector`: Check if two vectors are equal.
- `vector != vector`: Check if two vectors are not equal.

### Vector Expressions

Vector operations return lazy expressions that can be evaluated when needed. The following expression types are supported:

- `VectorAdd<T>`: Result of vector addition.
- `VectorSub<T>`: Result of vector subtraction.
- `VectorMul<T>`: Result of element-wise multiplication.

#### Methods for Vector Expressions

- `size()`: Get the number of elements.
- `operator[](i)`: Access the element at the specified index.
- `eval()`: Evaluate the expression and return a concrete Vector.

## Examples

### Matrix Examples

#### Example 1: Creating and Manipulating Matrices

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

#### Example 2: Matrix Dot Product

```python
import matrix_cpu

a = matrix_cpu.MatrixFloat(2, 3)
a.setConstant(2.0)

b = matrix_cpu.MatrixFloat(3, 2)
b.setConstant(3.0)

c = a * b
print("Matrix A (2x3):")
print(a)
print("Matrix B (3x2):")
print(b)
print("Matrix multiplication A·B (2x2):")
print(c)
```

#### Example 3: Matrix Operators (Addition, Subtraction, Multiplication)

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

### Vector Examples

#### Example 1: Creating and Manipulating Vectors

```python
import vector_cpu

# Create a vector with capacity 5
v = vector_cpu.VectorFloat(5)
v.resize(5, 2.5)  # Resize to 5 elements with value 2.5

print("Vector v:")
print(v)

# Compute the sum using reduce
import functools
import operator
sum_elements = functools.reduce(operator.add, [v[i] for i in range(v.size())], 0.0)
print(f"Sum of elements: {sum_elements}")  # Should be 12.5 for 5 elements of 2.5

# Access elements
print(f"First element: {v.front()}")
print(f"Last element: {v.back()}")
print(f"Element at index 2: {v[2]}")
```

#### Example 2: Vector Dot Product

```python
import vector_cpu

a = vector_cpu.VectorFloat(3)
a.resize(3, 2.0)

b = vector_cpu.VectorFloat(3)
b.resize(3, 3.0)

dot_product = a.dot(b)
print("Vector A:")
print(a)
print("Vector B:")
print(b)
print(f"Dot product A·B: {dot_product}")  # Should be 18.0 (2*3 + 2*3 + 2*3)
```

#### Example 3: Vector Operators (Addition, Subtraction, Multiplication)

```python
import vector_cpu as vc

# Create vectors
a = vc.VectorFloat(4)
b = vc.VectorFloat(4)

# Set values for vector a
for i in range(4):
    a.push_back(float(i))  # [0, 1, 2, 3]

# Set values for vector b
for i in range(4):
    b.push_back(float(i + 4))  # [4, 5, 6, 7]

print("Vector a:")
print(a)
print("Vector b:")
print(b)

# Addition
c_add = a + b
print("\nAddition (a + b):")
print(c_add)  # This is a lazy expression
print("Evaluated result:")
print(c_add.eval())  # Evaluate to get a concrete vector

# Subtraction
c_sub = a - b
print("\nSubtraction (a - b):")
print(c_sub)  # This is a lazy expression
print("Evaluated result:")
print(c_sub.eval())  # Evaluate to get a concrete vector

# Element-wise multiplication
c_mul = a * b
print("\nElement-wise multiplication (a * b):")
print(c_mul)  # This is a lazy expression
print("Evaluated result:")
print(c_mul.eval())  # Evaluate to get a concrete vector

# Scalar operations
print("\nScalar operations:")
print("a + 2.0:", a + 2.0)
print("3.0 * a:", 3.0 * a)
print("b - 1.0:", b - 1.0)
print("b / 2.0:", b / 2.0)
```

#### Example 4: Dynamic Vector Operations

```python
import vector_cpu as vc

# Create an empty vector
v = vc.VectorFloat()
print(f"Empty vector - Size: {v.size()}, Capacity: {v.capacity()}")

# Reserve space
v.reserve(10)
print(f"After reserve(10) - Size: {v.size()}, Capacity: {v.capacity()}")

# Add elements
for i in range(5):
    v.push_back(float(i * i))

print(f"After adding 5 elements - Size: {v.size()}, Capacity: {v.capacity()}")
print("Vector contents:", v)

# Modify elements
v[2] = 99.0
print("After modifying an element:", v)

# Pop back
v.pop_back()
print(f"After pop_back() - Size: {v.size()}, Vector: {v}")

# Clear
v.clear()
print(f"After clear() - Size: {v.size()}, Capacity: {v.capacity()}")

# Check if empty
print(f"Is empty? {v.empty()}")
```
