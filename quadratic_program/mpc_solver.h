#ifndef MPC_SOLVER_H
#define MPC_SOLVER_H

#include "math_utils.h"

// To connect to config file...
#ifndef MPC_HORIZON
#define MPC_HORIZON 48 // Default value, kinda arbitrary
#endif

template <int N = MPC_HORIZON>

class EMBEDDED_MPC_SOLVER {
    public:
        double MAX_THRUST_N;              // Max thrust (N)
        double rho;                     // Penalty param - 
        double sigma;                   // Regularization param - for numerical stability

        // Input variables
        // Flat 2D arrays, so row i, column j is [i * N + j]
        double H_inv[N * N];            // Inverse of Hessian matrix (N x N)
        double g[N];                    // Gradient vector 
        double W[N * N];                // Keep out tangennt normal matrix 
        double b[N];                    // Keep out distance vector

        // Solver variables - persist between calls to enable warm-starting
        double u[N];                    // Control vector 
        double z1[N];                   // Auxiliary variable -- thrust constraints
        double z2[N];                   // Auxiliary variable -- keep out constraints
        double y1[N];                   // Dual variable -- thrust constraints
        double y2[N];                   // Dual variable -- keep out constraints

        double S[N * N];                // Complement matrix for ADMM updates
        double L[N * N];                // Lower triangular fac of S

        // Constructor
        EMBEDDED_MPC_SOLVER(double max_thrust_in, double rho_in, double sigma_in){
            MAX_THRUST_N = max_thrust_in;
            rho = rho_in;
            sigma = sigma_in;

            // All vectors to zero
            for (int i = 0; i < N; i++){
                g[i] = 0.0;
                b[i] = 0.0;
                u[i] = 0.0;
                z1[i] = 0.0;
                z2[i] = 0.0;
                y1[i] = 0.0;
                y2[i] = 0.0;
            }

            // Matricies to zero as well
            for (int i = 0; i < N * N; i++){
                H_inv[i] = 0.0;
                W[i] = 0.0;
                S[i] = 0.0;
                L[i] = 0.0;
            }
        }

        // Solve method
        bool solve(int max_iterations){
            // Calculate S 
            double tempWH[N * N];

            // W * H_inv
            MATH::mat_mat_mul(W, H_inv, tempWH, N, N, N);

            // S = tempW * W^T
            MATH::mat_mat_trans_mul(tempWH, W, S, N, N, N);

            // S = S + 1/rho * I
            for (int i = 0; i < N; i++){
                S[i * N + i] += 1.0 / rho;
            }

            // Decomse for lower triangular L, S = L * L^T
            // if not positive-definite, must be false
            bool pos_def = MATH::cholesky_decompose(S, L, N);
            if (!pos_def){
                return false;
            }

            // --- ADMM iterations ---
            double prev_u[N];

        }

};

#endif // MPC_SOLVER_H