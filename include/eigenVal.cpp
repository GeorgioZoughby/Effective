#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// ======================== Template Matrix Class ========================
template <typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t rows, cols;

public:
    // Constructors
    Matrix(size_t rows, size_t cols, T val = T{}) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<T>(cols, val));
    }

    // Move constructor
    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(std::move(other.data)) {}

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Access operators
    std::vector<T>& operator[](size_t i) { return data[i]; }
    const std::vector<T>& operator[](size_t i) const { return data[i]; }

    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication!");
        }

        Matrix result(rows, other.cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                T sum = 0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T>& vec) const {
        if (cols != vec.size()) {
            throw std::invalid_argument("Matrix and vector dimensions mismatch!");
        }

        std::vector<T> result(rows);
        #pragma omp parallel for
        for (size_t i = 0; i < rows; ++i) {
            T sum = 0;
            for (size_t j = 0; j < cols; ++j) {
                sum += data[i][j] * vec[j];
            }
            result[i] = sum;
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j][i] = data[i][j];
            }
        }
        return result;
    }

    // Print matrix
    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ":\n";
        for (const auto& row : data) {
            for (T val : row) {
                std::cout << std::setw(12) << val << " ";
            }
            std::cout << "\n";
        }
    }
};

// ======================== Linear Algebra Utilities ========================
template <typename T>
T vector_norm(const std::vector<T>& vec) {
    T sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < vec.size(); ++i) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

template <typename T>
std::vector<T> normalize(const std::vector<T>& vec) {
    T n = vector_norm(vec);
    std::vector<T> normalized(vec.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); ++i) {
        normalized[i] = vec[i] / n;
    }
    return normalized;
}

// ======================== Power Iteration (Enhanced) ========================
template <typename T>
std::pair<T, std::vector<T>> power_iteration(const Matrix<T>& A, size_t max_iter = 1000, T tol = 1e-10) {
    size_t n = A.getRows();
    std::vector<T> b(n, 1.0); // Initial guess
    b = normalize(b);

    T eigenvalue = 0;
    size_t iter = 0;

    for (; iter < max_iter; ++iter) {
        std::vector<T> Ab = A * b;
        T new_eigenvalue = vector_norm(Ab);
        std::vector<T> new_b = normalize(Ab);

        // Check for convergence
        if (std::abs(new_eigenvalue - eigenvalue) < tol) {
            break;
        }

        eigenvalue = new_eigenvalue;
        b = new_b;
    }

    std::cout << "Power Iteration converged in " << iter << " iterations\n";
    return {eigenvalue, b};
}

// ======================== QR Decomposition (Householder) ========================
template <typename T>
void householder_qr(const Matrix<T>& A, Matrix<T>& Q, Matrix<T>& R) {
    size_t n = A.getRows();
    Q = Matrix<T>(n, n);
    R = A;

    for (size_t k = 0; k < n; ++k) {
        std::vector<T> x(n - k);
        for (size_t i = k; i < n; ++i) {
            x[i - k] = R[i][k];
        }

        T norm_x = vector_norm(x);
        T alpha = -std::copysign(norm_x, x[0]);
        x[0] -= alpha;
        norm_x = vector_norm(x);

        if (norm_x < 1e-12) continue;

        std::vector<T> v = normalize(x);

        // Apply Householder transformation to R
        for (size_t j = k; j < n; ++j) {
            T dot = 0;
            for (size_t i = k; i < n; ++i) {
                dot += v[i - k] * R[i][j];
            }
            for (size_t i = k; i < n; ++i) {
                R[i][j] -= 2 * v[i - k] * dot;
            }
        }

        // Accumulate Householder transformations into Q
        for (size_t j = 0; j < n; ++j) {
            T dot = 0;
            for (size_t i = k; i < n; ++i) {
                dot += Q[j][i] * v[i - k];
            }
            for (size_t i = k; i < n; ++i) {
                Q[j][i] -= 2 * dot * v[i - k];
            }
        }
    }

    Q = Q.transpose();
}

// ======================== QR Algorithm (With Shifts) ========================
template <typename T>
std::vector<T> qr_algorithm(Matrix<T> A, size_t max_iter = 500, T tol = 1e-12) {
    size_t n = A.getRows();
    std::vector<T> eigenvalues(n);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Wilkinson shift
        T a = A[n-2][n-2], b = A[n-2][n-1], c = A[n-1][n-2], d = A[n-1][n-1];
        T delta = (a - d) / 2;
        T shift = d - std::copysign(b * c, delta) / (std::abs(delta) + std::sqrt(delta * delta + b * c));

        // Subtract shift from diagonal
        for (size_t i = 0; i < n; ++i) {
            A[i][i] -= shift;
        }

        Matrix<T> Q(n, n), R(n, n);
        householder_qr(A, Q, R);
        A = R * Q;

        // Add shift back
        for (size_t i = 0; i < n; ++i) {
            A[i][i] += shift;
        }

        // Check for convergence
        bool converged = true;
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = A[i][i];
            for (size_t j = 0; j < i; ++j) {
                if (std::abs(A[i][j]) > tol) {
                    converged = false;
                }
            }
        }
        if (converged) break;
    }

    return eigenvalues;
}

#ifdef USE_CUDA
// ======================== CUDA Implementation ========================
void cuda_power_iteration(const Matrix<double>& A, double& eigenvalue, std::vector<double>& eigenvector) {
    // CUDA implementation would go here
    // This would use cuBLAS for matrix-vector operations
    // and custom kernels for vector normalization
    throw std::runtime_error("CUDA implementation not shown for brevity");
}
#endif

// ======================== Benchmarking ========================
template <typename Func, typename... Args>
auto benchmark(Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = std::forward<Func>(func)(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds\n";
    return result;
}

int main() {
    // Set up OpenMP
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "Using " << omp_get_max_threads() << " threads\n";

    // Example matrix (symmetric)
    Matrix<double> A(4, 4);
    A[0] = {4.0, 1.0, 1.0, 0.5};
    A[1] = {1.0, 3.0, 1.0, 0.5};
    A[2] = {1.0, 1.0, 2.0, 0.5};
    A[3] = {0.5, 0.5, 0.5, 1.0};
    A.print("Input Matrix");

    // Benchmark Power Iteration
    std::cout << "\n=== Power Iteration ===\n";
    auto [eigenvalue, eigenvector] = benchmark(power_iteration<double>, A);

    std::cout << "Dominant Eigenvalue: " << eigenvalue << "\n";
    std::cout << "Eigenvector: ";
    for (double val : eigenvector) std::cout << val << " ";
    std::cout << "\n";

    // Benchmark QR Algorithm
    std::cout << "\n=== QR Algorithm ===\n";
    auto eigenvalues = benchmark(qr_algorithm<double>, A);

    std::cout << "All Eigenvalues: ";
    for (double val : eigenvalues) std::cout << val << " ";
    std::cout << "\n";

#ifdef USE_CUDA
    // Benchmark CUDA Implementation
    std::cout << "\n=== CUDA Implementation ===\n";
    double cuda_eigenvalue;
    std::vector<double> cuda_eigenvector(A.getRows());
    benchmark([&]() { cuda_power_iteration(A, cuda_eigenvalue, cuda_eigenvector); });

    std::cout << "CUDA Dominant Eigenvalue: " << cuda_eigenvalue << "\n";
    std::cout << "CUDA Eigenvector: ";
    for (double val : cuda_eigenvector) std::cout << val << " ";
    std::cout << "\n";
#endif

    return 0;
}