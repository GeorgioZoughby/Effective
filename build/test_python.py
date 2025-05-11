import matrix_cpu
import time
import numpy as np

def main():
    # Create a Matrix of integers
    matrix_int = matrix_cpu.MatrixInt(30, 30)
    matrix_int.setOnes()
    print("Matrix of Integers (set to ones):")
    print(matrix_int)

    # Create a Matrix of floats
    matrix_float = matrix_cpu.MatrixFloat(20, 20)
    matrix_float.setConstant(3.14)
    print("\nMatrix of Floats (set to constant 3.14):")
    print(matrix_float)

    # Perform operations
    # print("\nSum of elements in float matrix:", matrix_float.sum())
    # print("Transpose of float matrix:")
    # print(matrix_float.transpose())

# Benchmarking function
def benchmark_matrix_operations():
    print("\nBenchmarking Matrix Operations:")

    # Benchmark for Matrix of integers
    start_time = time.time()
    matrix_int = matrix_cpu.MatrixInt(10000, 10000)
    matrix_int.setOnes()
    int_sum = matrix_int.sum()
    end_time = time.time()
    print(f"MatrixInt (10000x10000) sum: {int_sum}, Time taken: {end_time - start_time:.6f} seconds")

    # Benchmark for Matrix of floats
    start_time = time.time()
    matrix_float = matrix_cpu.MatrixFloat(10000, 10000)
    matrix_float.setConstant(3.14)
    float_sum = matrix_float.sum()
    end_time = time.time()
    print(f"MatrixFloat (10000x10000) sum: {float_sum}, Time taken: {end_time - start_time:.6f} seconds")

    # Benchmark for transpose operation
    start_time = time.time()
    # transpose_result = matrix_float.transpose()
    end_time = time.time()
    print(f"MatrixFloat (10000x10000) transpose, Time taken: {end_time - start_time:.6f} seconds")

    print("\nBenchmarking NumPy Matrix Operations:")

    # Benchmark for NumPy integer matrix
    start_time = time.time()
    numpy_int_matrix = np.ones((10000, 10000), dtype=int)
    numpy_int_sum = np.sum(numpy_int_matrix)
    end_time = time.time()
    print(f"NumPy MatrixInt (10000x10000) sum: {numpy_int_sum}, Time taken: {end_time - start_time:.6f} seconds")

    # Benchmark for NumPy float matrix
    start_time = time.time()
    numpy_float_matrix = np.full((10000, 10000), 3.14, dtype=float)
    numpy_float_sum = np.sum(numpy_float_matrix)
    end_time = time.time()
    print(f"NumPy MatrixFloat (10000x10000) sum: {numpy_float_sum}, Time taken: {end_time - start_time:.6f} seconds")

    # Benchmark for NumPy transpose operation
    start_time = time.time()
    numpy_transpose_result = np.transpose(numpy_float_matrix)
    end_time = time.time()
    print(f"NumPy MatrixFloat (10000x10000) transpose, Time taken: {end_time - start_time:.6f} seconds")

def raw_python_matrix_operations():
    print("\nBenchmarking Raw Python Matrix Operations:")

    # Benchmark for raw Python integer matrix
    start_time = time.time()
    python_int_matrix = [[1 for _ in range(10000)] for _ in range(10000)]
    python_int_sum = sum(sum(row) for row in python_int_matrix)
    end_time = time.time()
    print(f"Raw Python MatrixInt (10000x10000) sum: {python_int_sum}, Time taken: {end_time - start_time:.6f} seconds")

    # Benchmark for raw Python float matrix
    start_time = time.time()
    python_float_matrix = [[3.14 for _ in range(10000)] for _ in range(10000)]
    python_float_sum = sum(sum(row) for row in python_float_matrix)
    end_time = time.time()
    print(f"Raw Python MatrixFloat (10000x10000) sum: {python_float_sum}, Time taken: {end_time - start_time:.6f} seconds")

    # Benchmark for raw Python transpose operation
    start_time = time.time()
    python_transpose_result = list(map(list, zip(*python_float_matrix)))
    end_time = time.time()
    print(f"Raw Python MatrixFloat (10000x10000) transpose, Time taken: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()
    benchmark_matrix_operations()
    raw_python_matrix_operations()