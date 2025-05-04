#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>

template <typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t rows, cols;

public:
    Matrix(size_t rows, size_t cols, T val = T{}) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<T>(cols, val));
    }

    // Access operators
    std::vector<T>& operator[](size_t i) { return data[i]; }
    const std::vector<T>& operator[](size_t i) const { return data[i]; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ":\n";
        for (const auto& row : data) {
            for (T val : row) std::cout << std::setw(12) << val << " ";
            std::cout << "\n";
        }
    }
};

// ======================== LU Decomposition ========================
template <typename T>
void lu_decomposition(const Matrix<T>& A, Matrix<T>& L, Matrix<T>& U) {
    size_t n = A.getRows();
    L = Matrix<T>(n, n);
    U = Matrix<T>(n, n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        L[i][i] = 1.0; // Diagonal of L is 1

        for (size_t j = 0; j < n; ++j) {
            if (j >= i) { // Upper triangle + diagonal
                T sum = 0.0;
                for (size_t k = 0; k < i; ++k) {
                    sum += L[i][k] * U[k][j];
                }
                U[i][j] = A[i][j] - sum;
            }

            if (j < i) { // Lower triangle
                T sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += L[i][k] * U[k][j];
                }
                L[i][j] = (A[i][j] - sum) / U[j][j];
            }
        }
    }
}

// ======================== QR Decomposition (Householder) ========================
template <typename T>
void qr_decomposition(const Matrix<T>& A, Matrix<T>& Q, Matrix<T>& R) {
    size_t m = A.getRows();
    size_t n = A.getCols();
    Q = Matrix<T>(m, m);
    R = A;

    std::vector<Matrix<T>> reflectors;

    for (size_t k = 0; k < n; ++k) {
        std::vector<T> x(m - k);
        for (size_t i = k; i < m; ++i) {
            x[i - k] = R[i][k];
        }

        T norm_x = 0.0;
        #pragma omp parallel for reduction(+:norm_x)
        for (size_t i = 0; i < x.size(); ++i) {
            norm_x += x[i] * x[i];
        }
        norm_x = std::sqrt(norm_x);

        T alpha = -std::copysign(norm_x, x[0]);
        x[0] -= alpha;
        T norm_v = std::sqrt(norm_x * (norm_x + std::abs(x[0])));

        if (norm_v < std::numeric_limits<T>::epsilon()) continue;

        #pragma omp parallel for
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] /= norm_v;
        }

        // Apply Householder to R
        for (size_t j = k; j < n; ++j) {
            T dot = 0.0;
            #pragma omp parallel for reduction(+:dot)
            for (size_t i = k; i < m; ++i) {
                dot += x[i - k] * R[i][j];
            }
            #pragma omp parallel for
            for (size_t i = k; i < m; ++i) {
                R[i][j] -= 2 * x[i - k] * dot;
            }
        }

        // Store reflector for Q construction
        Matrix<T> P(m, m);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                P[i][j] = (i == j) ? 1.0 : 0.0;
                if (i >= k && j >= k) {
                    P[i][j] -= 2 * x[i - k] * x[j - k];
                }
            }
        }
        reflectors.push_back(P);
    }

    // Build Q from reflectors
    Q = Matrix<T>(m, m);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            Q[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (const auto& P : reflectors) {
        Q = Q * P;
    }
}

// ======================== SVD (Simplified via Eigen Decomposition) ========================
template <typename T>
void svd(const Matrix<T>& A, Matrix<T>& U, std::vector<T>& S, Matrix<T>& Vt) {
    size_t m = A.getRows();
    size_t n = A.getCols();
    
    // Compute A^T * A for V and S^2
    Matrix<T> AtA(n, n);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0.0;
            for (size_t k = 0; k < m; ++k) {
                sum += A[k][i] * A[k][j];
            }
            AtA[i][j] = sum;
        }
    }

    // Eigen decomposition of A^T * A (simplified)
    Matrix<T> V(n, n);
    S.resize(std::min(m, n));
    // In practice, use QR algorithm or specialized SVD solver here
    // This is a placeholder for the actual SVD computation
    // For production code, consider using LAPACK or Eigen library

    // For U, compute A * V * Sigma^+
    // (Omitted for brevity - would use similar matrix operations)
}

int main() {
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n";

    Matrix<double> A(3, 3);
    A[0] = {4.0, 1.0, 1.0};
    A[1] = {1.0, 3.0, 1.0};
    A[2] = {1.0, 1.0, 2.0};
    A.print("Input Matrix A");

    // LU Decomposition
    Matrix<double> L(3, 3), U(3, 3);
    lu_decomposition(A, L, U);
    L.print("\nL Matrix");
    U.print("U Matrix");

    // QR Decomposition
    Matrix<double> Q(3, 3), R(3, 3);
    qr_decomposition(A, Q, R);
    Q.print("\nQ Matrix");
    R.print("R Matrix");

    // SVD (Simplified)
    Matrix<double> U_svd(3, 3), Vt(3, 3);
    std::vector<double> S;
    svd(A, U_svd, S, Vt);
    std::cout << "\nSingular Values: ";
    for (double s : S) std::cout << s << " ";
    std::cout << "\n";

    return 0;
}