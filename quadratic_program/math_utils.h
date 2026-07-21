#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <algorithm>
#include <cmath>

namespace MATH {

// --- Vector Math ---

// Vector addition
inline void vec_add(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

// Vector subtraction a - b
inline void vec_subtract(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}

// Vector scale and add: result = a + scalar * b
inline void vec_saxpy(const double* a, double scalar, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + scalar * b[i];
    }
}

// clamps elements of in between min_val and max_val
inline void vec_clip(const double* in, double min_val, double max_val, double* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = std::max(min_val, std::min(max_val, in[i]));
    }
}

// max(a, b)
inline void vec_max(const double* a, const double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = std::max(a[i], b[i]);
    }
}

// norm 
inline double vec_norm(const double* v, int size) {
    double sum_sq = 0.0;
    for (int i = 0; i < size; i++) {
        sum_sq += v[i] * v[i];
    }
    return std::sqrt(sum_sq);
}


// --- Matrix Math ---

// multiplication
// (rows x cols)
inline void mat_vec_mul(const double* A, const double* x, double* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        out[i] = sum;
    }
}

// transpose A^T * x
// A has dimensions (rows x cols)
inline void mat_trans_vec_mul(const double* A, const double* x, double* out, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += A[i * cols + j] * x[i];
        }
        out[j] = sum;
    }
}

//  multiplication: A * B
// A is (m x n), B is (n x p), C is (m x p)
inline void mat_mat_mul(const double* A, const double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// transpose multiplication: A * B^T
// A is (m x n), B is (p x n) [B^T is n x p], C is (m x p)
inline void mat_mat_trans_mul(const double* A, const double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[j * n + k];
            }
            C[i * p + j] = sum;
        }
    }
}


// --- Cholesky Solver (For Sherman-Morrison-Woodbury System) ---

/**
 * @brief Computes Cholesky decomposition of a symmetric positive-definite matrix: A = L * L^T
 * Writes lower-triangular L (row-major flat pointer). Returns false if not positive-definite.
 */
inline bool cholesky_decompose(const double* A, double* L, int n) {
    for (int i = 0; i < n * n; i++) {
        L[i] = 0.0;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                double val = A[i * n + i] - sum;
                if (val <= 0.0) {
                    return false; // Matrix is not positive-definite
                }
                L[i * n + j] = std::sqrt(val);
            } else {
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return true;
}

/**
 * @brief Solves L * L^T * x = b using forward/backward substitutions.
 * L is an (n x n) lower-triangular matrix, b is the rhs, x is the output.
 */
inline void cholesky_solve(const double* L, const double* b, double* x, int n) {
    // Stack-allocated temporary variable for forward substitution
    // Max supported horizon size is 120, which covers N=80 easily
    double y[120] = {0.0};
    
    // 1. Forward substitution: L * y = b
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < i; k++) {
            sum += L[i * n + k] * y[k];
        }
        y[i] = (b[i] - sum) / L[i * n + i];
    }
    
    // 2. Backward substitution: L^T * x = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int k = i + 1; k < n; k++) {
            sum += L[k * n + i] * x[k];
        }
        x[i] = (y[i] - sum) / L[i * n + i];
    }
}

} // namespace MATH
#endif // MATH_UTILS_H