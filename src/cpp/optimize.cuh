#pragma once
#include <cstddef>
#include <cstdint>

#include "cu3d/cub_wrapper.cuh"
#include "cu3d/cublas_wrapper.cuh"
#include "cu3d/cusolver_wrapper.cuh"
#include "cu3d/cusparse_wrapper.cuh"
#include "cu3d/gpu_structs.cuh"

namespace cu3d {

enum class SolverType : std::uint8_t { CGLS, LSQR, QR, LSMR };

void weightIndividualEquations(GPUFloatData& A, size_t rows, size_t columns,
                               const GPUFloatData& weight);

/** Solves a linear system of equations with a Jacobi Preconditioner.
 *
 * Arguments:
 * cublas        A CUBLASWrapper instance used to do linear algebra.
 * cub           A CUBWrapper instance for segmented reductions.
 * solver        The SolverType used to solve the preconditioned LSE.
 * A             Dense [A_n_rows, A_n_columns] matrix in column major order.
 * A_n_rows      Number of rows in A.
 * A_n_columns   Number of columns in A.
 * b             Vector of knowns in Ax=b. Shape [A_n_rows];
 * max_iter      Maximum number of iterations to use for the solver. Default =
 * 100. tol           Numeric tolerance for result for early stopping. Default =
 * 1e-6.
 */
GPUFloatData JacobiPreconditionedSolver(
    CUBLASWrapper& cublas, CUBWrapper& cub, CUSOLVERWrapper& cusolver,
    SolverType solver, const GPUFloatData& A, int64_t A_n_rows,
    int64_t A_n_columns, const GPUFloatData& b, const size_t max_iter = 100,
    const float tol = 1e-6);

/** Solves a sparse linear system of equations with a Jacobi Preconditioner.
 *
 * Arguments:
 * cublas: A CUBLASWrapper instance used to do linear algebra.
 * cub : A CUBWrapper instance for segmented reductions.
 * cusparse: A CUSPARSEWrapper instance used to do linear algebra.
 * solver: The SolverType used to solve the preconditioned LSE.
 * A_csr_row_offsets: The row offsets of A in the CSR format.
 * A_csr_column_idxs: The colujn indices of A in the CSR format.
 * A_values: The values of A in CSR format.
 * A_n_rows: Number of rows in A.
 * A_n_columns: Number of columns in A.
 * b: Vector of knowns in Ax=b. Shape [A_n_rows];
 * max_iter: Maximum number of iterations to use for the solver. Default = 100.
 * tol: Numeric tolerance for result for early stopping. Default = 1e-6.
 */
GPUFloatData JacobiPreconditionedSolver_sparse(
    CUBLASWrapper& cublas, CUSPARSEWrapper& cusparse, CUBWrapper& cub,
    SolverType solver, const GPUIntData& A_csr_row_offsets,
    const GPUIntData& A_csr_column_idxs, const GPUFloatData& A_csr_values,
    int64_t A_n_rows, int64_t A_n_columns, const GPUFloatData& b,
    const size_t max_iter = 100, const float tol = 1e-6);

/** Conjugate Gradient Least Squares algorithm
 * Solves the sparse linear system of equations Ax=b in a least squares sense.
 * Is equivalent of using Conjugate Gradient on the normal equation
 * ATAx = ATb.
 *
 * Arguments:
 * cublas               A CUBLASWrapper instance used to do linear algebra.
 * cusparse             A CUSPARSEWrapper instance used to do sparse linear
 *                      algebra.
 * A_csr_row_offsets    The row offsets of A in the CSR
 *                      format.
 * A_csr_column_idxs    The colujn indices of A in the CSR format.
 * A_values             The values of A in CSR format.
 * A_n_rows             number of rows in matrix A.
 * A_n_columns          number of columns in matrix A.
 * b                    Vector of size [A_n_rows].
 * max_iter             Maximum number of iterations for the algorithm.
 * tol                  Stopping criterion. If the gradient at the current x
 *                      is smaller than this x is returned.
 *
 * Returns:
 * The least squares solution vector x of shape [A_n_columns].
 */
GPUFloatData CGLS_sparse(CUBLASWrapper& cublas, CUSPARSEWrapper& cusparse,
                         const GPUIntData& A_csr_row_offsets,
                         const GPUIntData& A_csr_column_idxs,
                         const GPUFloatData& A_csr_values, int64_t A_n_rows,
                         int64_t A_n_columns, const GPUFloatData& b,
                         const size_t max_iter = 100, const float tol = 1e-4);

/** Conjugate Gradient Least Squares algorithm
 * Solves the linear system of equations Ax=b in a least squares sense.
 * Is equivalent of using Conjugate Gradient on the normal equation
 * ATAx = ATb.
 * For sparce matrices A!
 *
 * Arguments:
 * cublas        A CUBLASWrapper instance used to do linear algebra.
 * A             Dense [A_n_rows, A_n_columns] matrix in column major order.
 * A_n_rows      number of rows in matrix A.
 * A_n_columns   number of columns in matrix A.
 * b             Vector of size [A_n_rows].
 * max_iter      Maximum number of iterations for the algorithm.
 * tol           Stopping criterion. If the gradient at the current x is
 *               smaller than this x is returned.
 *
 * Returns:
 * The least squares solution vector x of shape [A_n_columns].
 */
GPUFloatData CGLS(CUBLASWrapper& cublas, const GPUFloatData& A,
                  int64_t A_n_rows, int64_t A_n_columns, const GPUFloatData& b,
                  const size_t max_iter = 100, const float tol = 1e-4);

/** LSQR algorithm for Least Squares problems
 * Solves the linear system of equations Ax=b in a least squares sense.
 * Is slightly more expensive than CGLS but is numerically more stable.
 *
 * Arguments:
 * cublas        A CUBLASWrapper instance used to do linear algebra.
 * A             Dense [A_n_rows, A_n_columns] matrix in column major order.
 * A_n_rows      number of rows in matrix A.
 * A_n_columns   number of columns in matrix A.
 * b             Vector of size [A_n_rows].
 * max_iter      Maximum number of iterations for the algorithm.
 * tol           Numeric tolerance used to evaluate the stopping criterions.
 *               ATOL = tol and BTOL = tol.
 * CONDLIM       Tolerance for the condition number of the internally
 *               solved matrix. If the condition number gets larger the
 *               optimization stops.
 *
 * Returns:
 * The least squares solution vector x of shape [A_n_columns].
 */
GPUFloatData LSQR(CUBLASWrapper& cublas, const GPUFloatData& A,
                  int64_t A_n_rows, int64_t A_n_columns, const GPUFloatData& b,
                  const size_t max_iter = 100, const float tol = 1e-4,
                  const float CONDLIM = 1e8f);

/** LSQR algorithm for Least Squares problems
 * Solves the linear system of equations Ax=b in a least squares sense.
 * Is slightly more expensive than CGLS but is numerically more stable.
 * For sparse matrices A!
 *
 * Arguments:
 * cublas               A CUBLASWrapper instance used to do linear algebra.
 * cusparse             A CUSPARSEWrapper instance used to do sparse linear
 *                      algebra.
 * A_csr_row_offsets    The row offsets of A in the CSR format.
 * A_csr_column_idxs    The colujn indices of A in the CSR format.
 * A_values             The values of A in CSR format.
 * A_n_rows             number of rows in matrix A.
 * A_n_columns          number of columns in matrix A.
 * b                    Vector of size [A_n_rows].
 * max_iter             Maximum number of iterations for the algorithm.
 * tol                  Numeric tolerance used to evaluate the stopping
 * criterions. ATOL = tol and BTOL = tol. CONDLIM              Tolerance for the
 * condition number of the internally solved matrix. If the condition number
 * gets larger the optimization stops.
 *
 * Returns:
 * The least squares solution vector x of shape [A_n_columns].
 */
GPUFloatData LSQR_sparse(CUBLASWrapper& cublas, CUSPARSEWrapper& cusparse,
                         const GPUIntData& A_csr_row_offsets,
                         const GPUIntData& A_csr_column_idxs,
                         const GPUFloatData& A_csr_values, int64_t A_n_rows,
                         int64_t A_n_columns, const GPUFloatData& b,
                         const size_t max_iter = 100, const float tol = 1e-4,
                         const float CONDLIM = 1e8f);

/** LSMR algorithm for Least Squares problems
 * Solves the linear system of equations Ax=b in a least squares sense.
 * Is slightly more expensive than CGLS but is numerically more stable.
 *
 * Arguments:
 * cublas        A CUBLASWrapper instance used to do linear algebra.
 * A             Dense [A_n_rows, A_n_columns] matrix in column major order.
 * A_n_rows      number of rows in matrix A.
 * A_n_columns   number of columns in matrix A.
 * b             Vector of size [A_n_rows].
 * dampening     Regularization factor (lambda in the paper)
 * max_iter      Maximum number of iterations for the algorithm.
 * tol           Numeric tolerance used to evaluate the stopping criterions.
 *               ATOL = tol and BTOL = tol.
 * CONDLIM       Tolerance for the condition number of the internally
 *               solved matrix. If the condition number gets larger the
 *               optimization stops.
 *
 * Returns:
 * The least squares solution vector x of shape [A_n_columns].
 */
GPUFloatData LSMR(CUBLASWrapper& cublas, const GPUFloatData& A,
                  int64_t A_n_rows, int64_t A_n_columns, const GPUFloatData& b,
                  float dampening = 0.f, const size_t max_iter = 100,
                  const float tol = 1e-4, const float CONDLIM = 1e8f);

/** LSMR algorithm for Least Squares problems
 * Solves the linear system of equations Ax=b in a least squares sense.
 * Is slightly more expensive than CGLS but is numerically more stable.
 * For sparse matrices A!
 *
 * Arguments:
 * cublas        A CUBLASWrapper instance used to do linear algebra.
 * cusparse      A CUSPARSEWrapper instance used to do sparse linear algebra.
 * A_csr_row_offsets    The row offsets of A in the CSR format.
 * A_csr_column_idxs    The colujn indices of A in the CSR format.
 * A_values             The values of A in CSR format.
 * A_n_rows      number of rows in matrix A.
 * A_n_columns   number of columns in matrix A.
 * b             Vector of size [A_n_rows].
 * dampening     Regularization factor (lambda in the paper)
 * max_iter      Maximum number of iterations for the algorithm.
 * tol           Numeric tolerance used to evaluate the stopping criterions.
 *               ATOL = tol and BTOL = tol.
 * CONDLIM       Tolerance for the condition number of the internally
 *               solved matrix. If the condition number gets larger the
 *               optimization stops.
 *
 * Returns:
 * The least squares solution vector x of shape [A_n_columns].
 */
GPUFloatData LSMR_sparse(CUBLASWrapper& cublas, CUSPARSEWrapper& cusparse,
                         const GPUIntData& A_csr_row_offsets,
                         const GPUIntData& A_csr_column_idxs,
                         const GPUFloatData& A_csr_values, int64_t A_n_rows,
                         int64_t A_n_columns, const GPUFloatData& b,
                         float dampening = 0.f, const size_t max_iter = 100,
                         const float tol = 1e-4, const float CONDLIM = 1e8f);

GPUFloatData directQR(CUSOLVERWrapper& cusolver, const GPUFloatData& A,
                      size_t rows, size_t columns, const GPUFloatData& b);

}  // namespace cu3d
