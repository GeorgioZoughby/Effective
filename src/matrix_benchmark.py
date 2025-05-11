#!/usr/bin/env python3
import sys
import ctypes
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import random
cuda_dll_path = r".\cudart64_12.dll"
python_dll_path = r".\python39.dll"  # Replace with your Python installation path

# Add the directories containing the DLLs to the system path
os.environ["PATH"] += os.pathsep + os.path.dirname(cuda_dll_path)
os.environ["PATH"] += os.pathsep + os.path.dirname(python_dll_path)

# Now attempt to load the DLLs manually
cuda_dll = ctypes.CDLL(cuda_dll_path)
python_dll = ctypes.CDLL(python_dll_path)

print("DLLs loaded successfully.")
# Add parent directory to path to import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the build directory containing the matrix_gpu.pyd file to the system path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'gpu', 'Debug'))
import matrix_gpu
import vector_gpu
import time
import numpy as np

plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (12, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def format_time(time_ns, pos):
    """Format time in ms with appropriate precision"""
    if time_ns >= 1_000_000:  # If over 1ms
        return f'{time_ns / 1_000_000:.1f} ms'
    else:
        return f'{time_ns / 1_000:.1f} µs'


def time_function(func, *args, **kwargs):
    """Measure execution time of a function in nanoseconds"""
    # Warm up
    func(*args, **kwargs)

    # Actual timing
    iterations = kwargs.pop('iterations', 10)
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        times.append(end - start)

    return min(times), result  # Return the best time and the result


# Helper functions for creating test data
def create_random_list(n):
    """Create a list of n random floats"""
    return [random.random() for _ in range(n)]


def create_random_2d_list(rows, cols):
    """Create a 2D list of random floats"""
    return [[random.random() for _ in range(cols)] for _ in range(rows)]


def create_custom_vector(values):
    """Create a custom Vector from list of values"""
    v = vector_gpu.VectorFloat()
    for val in values:
        v.push_back(float(val))
    return v


def create_custom_matrix(values):
    """Create a custom Matrix from 2D list of values"""
    rows = len(values)
    cols = len(values[0]) if rows > 0 else 0
    m = matrix_gpu.MatrixFloat(rows, cols)
    for i in range(rows):
        for j in range(cols):
            m.set(i, j, float(values[i][j]))
    return m


class VectorBenchmarks:
    @staticmethod
    def setup(sizes):
        results = {}
        for n in sizes:
            # Generate data
            data_a = create_random_list(n)
            data_b = create_random_list(n)

            # Create different representations
            vec_gpu_a = create_custom_vector(data_a)
            vec_gpu_b = create_custom_vector(data_b)
            np_a = np.array(data_a, dtype=np.float32)
            np_b = np.array(data_b, dtype=np.float32)

            results[n] = {
                'data_a': data_a,
                'data_b': data_b,
                'vec_gpu_a': vec_gpu_a,
                'vec_gpu_b': vec_gpu_b,
                'np_a': np_a,
                'np_b': np_b,
            }
        return results

    @staticmethod
    def vector_addition():
        def numpy_add(a, b):
            result = a + b
            return result[0]  # Access element to ensure computation is done

        def custom_add(a, b):
            result = a + b
            dummy = result[0]  # Force evaluation
            return result

        def python_add(a, b):
            return [a[i] + b[i] for i in range(len(a))]

        return {
            'numpy': numpy_add,
            'custom': custom_add,
            'python': python_add
        }

    @staticmethod
    def vector_dot_product():
        def numpy_dot(a, b):
            return np.dot(a, b)

        def custom_dot(a, b):
            return a.dot(b)

        def python_dot(a, b):
            return sum(a[i] * b[i] for i in range(len(a)))

        return {
            'numpy': numpy_dot,
            'custom': custom_dot,
            'python': python_dot
        }

    @staticmethod
    def vector_scalar_mult():
        scalar = 2.5

        def numpy_scalar_mult(a, _):
            result = a * scalar
            return result[0]  # Access element to ensure computation is done

        def custom_scalar_mult(a, _):
            result = a * scalar
            dummy = result[0]  # Force evaluation
            return result

        def python_scalar_mult(a, _):
            return [x * scalar for x in a]

        return {
            'numpy': numpy_scalar_mult,
            'custom': custom_scalar_mult,
            'python': python_scalar_mult
        }


class MatrixBenchmarks:
    @staticmethod
    def setup(sizes):
        results = {}
        for n in sizes:
            # Generate data
            data_a = create_random_2d_list(n, n)
            data_b = create_random_2d_list(n, n)

            # Create different representations
            mat_cpu_a = create_custom_matrix(data_a)
            mat_cpu_b = create_custom_matrix(data_b)
            np_a = np.array(data_a, dtype=np.float32)
            np_b = np.array(data_b, dtype=np.float32)

            results[n] = {
                'data_a': data_a,
                'data_b': data_b,
                'mat_cpu_a': mat_cpu_a,
                'mat_cpu_b': mat_cpu_b,
                'np_a': np_a,
                'np_b': np_b,
            }
        return results

    @staticmethod
    def matrix_multiplication():
        def numpy_mult(a, b):
            return np.matmul(a, b)

        def custom_mult(a, b):
            result = a * b
            return result.eval()  # Need to evaluate the lazy expression

        def python_mult(a, b):
            n = len(a)
            result = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i][j] += a[i][k] * b[k][j]
            return result

        return {
            'numpy': numpy_mult,
            'custom': custom_mult,
            'python': python_mult
        }

    @staticmethod
    def matrix_transpose():
        def numpy_transpose(a, _):
            return a.T

        def custom_transpose(a, _):
            return a.transpose()  # Directly return the result of transpose

        def python_transpose(a, _):
            n = len(a)
            result = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    result[j][i] = a[i][j]
            return result

        return {
            'numpy': numpy_transpose,
            'custom': custom_transpose,
            'python': python_transpose
        }


def run_vector_benchmarks():
    sizes = [100, 1000, 10000, 100000, 500000, 1000000]
    operations = [
        ("Vector Dot Product", VectorBenchmarks.vector_dot_product()),
        ("Vector Scalar Multiplication", VectorBenchmarks.vector_scalar_mult()),
    ]

    data = VectorBenchmarks.setup(sizes)
    results = {}

    for op_name, op_funcs in operations:
        print(f"Benchmarking {op_name}...")
        results[op_name] = {size: {'numpy': None, 'custom': None, 'python': None} for size in sizes}

        for size in sizes:
            test_data = data[size]

            # NumPy version
            numpy_time, _ = time_function(
                op_funcs['numpy'],
                test_data['np_a'],
                test_data['np_b']
            )
            results[op_name][size]['numpy'] = numpy_time

            # Custom version
            custom_time, _ = time_function(
                op_funcs['custom'],
                test_data['vec_gpu_a'],
                test_data['vec_gpu_b']
            )
            results[op_name][size]['custom'] = custom_time

            # Python version is too slow for very large vectors
            if size <= 100000:
                python_time, _ = time_function(
                    op_funcs['python'],
                    test_data['data_a'],
                    test_data['data_b']
                )
                results[op_name][size]['python'] = python_time

    return results, sizes


def run_matrix_benchmarks():
    # Added larger matrix sizes and more granularity
    sizes = [10, 50, 100, 200, 500, 1000]
    operations = [
        ("Matrix Multiplication", MatrixBenchmarks.matrix_multiplication()),
        ("Matrix Transpose", MatrixBenchmarks.matrix_transpose())
    ]

    data = MatrixBenchmarks.setup(sizes)
    results = {}

    for op_name, op_funcs in operations:
        print(f"Benchmarking {op_name}...")
        results[op_name] = {size: {'numpy': None, 'custom': None, 'python': None} for size in sizes}

        for size in sizes:
            test_data = data[size]

            # NumPy version
            numpy_time, _ = time_function(
                op_funcs['numpy'],
                test_data['np_a'],
                test_data['np_b']
            )
            results[op_name][size]['numpy'] = numpy_time

            # Custom version - only benchmark reasonable sizes
            if size <= 500:  # Skip very large matrices for custom implementation if too slow
                custom_time, _ = time_function(
                    op_funcs['custom'],
                    test_data['mat_cpu_a'],
                    test_data['mat_cpu_b']
                )
                results[op_name][size]['custom'] = custom_time

            # For large matrices, Python can be very slow, so we'll limit the size
            if size <= 50:  # Only measure Python implementation for small matrices
                python_time, _ = time_function(
                    op_funcs['python'],
                    test_data['data_a'],
                    test_data['data_b']
                )
                results[op_name][size]['python'] = python_time

    return results, sizes


def plot_results(vector_results, vector_sizes, matrix_results, matrix_sizes):
    # Calculate total number of subplots needed
    total_plots = len(vector_results) + len(matrix_results)
    
    # Create a figure with the appropriate number of subplots
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 5 * total_plots))
    
    # Ensure axes is always a list/array for consistent handling
    if total_plots == 1:
        axes = [axes]
    
    # Plot vector operations
    for i, (op_name, size_results) in enumerate(vector_results.items()):
        ax = axes[i]
        
        numpy_times = []
        custom_times = []
        python_times = []
        
        for size in vector_sizes:
            numpy_times.append(size_results[size]['numpy'])
            
            custom_time = size_results[size].get('custom')
            custom_times.append(custom_time if custom_time is not None else float('nan'))
            
            python_time = size_results[size].get('python')
            python_times.append(python_time if python_time is not None else float('nan'))
        
        x = range(len(vector_sizes))
        width = 0.25
        
        ax.bar([p - width for p in x], numpy_times, width, label='NumPy', color='#1f77b4')
        ax.bar(x, custom_times, width, label='Custom GPU', color='#ff7f0e')
        ax.bar([p + width for p in x], python_times, width, label='Python', color='#2ca02c')
        
        ax.set_title(f"{op_name} Performance")
        ax.set_xticks(x)
        ax.set_xticklabels([f'{size:,}' for size in vector_sizes])
        ax.set_xlabel('Vector Size')
        ax.set_ylabel('Time')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(format_time))
        ax.legend()
        
        # Add time values above bars
        for j, (numpy_t, custom_t, python_t) in enumerate(zip(numpy_times, custom_times, python_times)):
            if numpy_t > 0:  # Only add label if time is positive
                ax.text(j - width, numpy_t * 1.05, format_time(numpy_t, None), ha='center', va='bottom', rotation=45, fontsize=8)
            if not np.isnan(custom_t) and custom_t > 0:
                ax.text(j, custom_t * 1.05, format_time(custom_t, None), ha='center', va='bottom', rotation=45, fontsize=8)
            if not np.isnan(python_t) and python_t > 0:
                ax.text(j + width, python_t * 1.05, format_time(python_t, None), ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Plot matrix operations
    for i, (op_name, size_results) in enumerate(matrix_results.items()):
        ax = axes[len(vector_results) + i]
        
        numpy_times = []
        custom_times = []
        python_times = []
        
        for size in matrix_sizes:
            numpy_times.append(size_results[size]['numpy'])
            
            custom_time = size_results[size].get('custom')
            custom_times.append(custom_time if custom_time is not None else float('nan'))
            
            python_time = size_results[size].get('python')
            python_times.append(python_time if python_time is not None else float('nan'))
        
        x = range(len(matrix_sizes))
        width = 0.25
        
        ax.bar([p - width for p in x], numpy_times, width, label='NumPy', color='#1f77b4')
        ax.bar(x, custom_times, width, label='Custom GPU', color='#ff7f0e')
        ax.bar([p + width for p in x], python_times, width, label='Python', color='#2ca02c')
        
        ax.set_title(f"{op_name} Performance")
        ax.set_xticks(x)
        ax.set_xticklabels([f'{size}x{size}' for size in matrix_sizes])
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Time')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(format_time))
        ax.legend()
        
        # Add time values above bars
        for j, (numpy_t, custom_t, python_t) in enumerate(zip(numpy_times, custom_times, python_times)):
            if numpy_t > 0:  # Only add label if time is positive
                ax.text(j - width, numpy_t * 1.05, format_time(numpy_t, None), ha='center', va='bottom', rotation=45, fontsize=8)
            if not np.isnan(custom_t) and custom_t > 0:
                ax.text(j, custom_t * 1.05, format_time(custom_t, None), ha='center', va='bottom', rotation=45, fontsize=8)
            if not np.isnan(python_t) and python_t > 0:
                ax.text(j + width, python_t * 1.05, format_time(python_t, None), ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'performance_comparison.png'")
    plt.show()


def main():
    print("Running vector benchmarks...")
    vector_results, vector_sizes = run_vector_benchmarks()

    print("\nRunning matrix benchmarks...")
    matrix_results, matrix_sizes = run_matrix_benchmarks()

    print("\nGenerating plots...")
    plot_results(vector_results, vector_sizes, matrix_results, matrix_sizes)

    # Print summary
    print("\nPerformance Summary:")
    print("===================")

    print("\nVector Operations:")
    for op_name, size_results in vector_results.items():
        print(f"\n  {op_name}:")
        for size in vector_sizes:
            numpy_time = size_results[size]['numpy']
            custom_time = size_results[size]['custom']
            python_time = size_results[size].get('python')

            numpy_vs_custom = custom_time / numpy_time

            print(f"    Size {size:,}:")
            print(f"      NumPy: {format_time(numpy_time, None)}")
            print(f"      Custom C++: {format_time(custom_time, None)} ({numpy_vs_custom:.2f}x NumPy)")

            if python_time is not None:
                numpy_vs_python = python_time / numpy_time
                print(f"      Python: {format_time(python_time, None)} ({numpy_vs_python:.2f}x NumPy)")
            else:
                print(f"      Python: Not measured (too slow)")

    print("\nMatrix Operations:")
    for op_name, size_results in matrix_results.items():
        print(f"\n  {op_name}:")
        for size in matrix_sizes:
            numpy_time = size_results[size]['numpy']
            custom_time = size_results[size].get('custom')
            python_time = size_results[size].get('python')

            print(f"    Size {size}x{size}:")
            print(f"      NumPy: {format_time(numpy_time, None)}")

            if custom_time is not None:
                numpy_vs_custom = custom_time / numpy_time
                print(f"      Custom C++: {format_time(custom_time, None)} ({numpy_vs_custom:.2f}x NumPy)")
            else:
                print(f"      Custom C++: Not measured (too slow)")

            if python_time is not None:
                numpy_vs_python = python_time / numpy_time
                print(f"      Python: {format_time(python_time, None)} ({numpy_vs_python:.2f}x NumPy)")
            else:
                print(f"      Python: Not measured (too slow)")


if __name__ == "__main__":
    main()