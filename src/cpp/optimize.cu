#include <glog/logging.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "cu3d/cub_wrapper.cuh"
#include "cu3d/cusolver_wrapper.cuh"
#include "cu3d/cusparse_wrapper.cuh"
#include "cu3d/device_inline.cuh"
#include "cu3d/gpu_structs.cuh"
#include "cu3d/optimize.cuh"

namespace cu3d {

__global__ void weightRows(GPUFloatData::Reference A, uint32_t rows,
                           uint32_t columns,
                           GPUFloatData::ConstReference row_weights) {
  // Note: A is column major!
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= A.num_elements) {
    return;
  }
  uint32_t row_id = tid / columns;
  uint32_t column_id = tid % columns;
  float weight = row_weights.data[row_id];
  A.data[row_id + column_id * rows] *= weight;
}

void weightIndividualEquations(GPUFloatData& A, size_t rows, size_t columns,
                               const GPUFloatData& weight) {
  CHECK_EQ(rows, weight.size) << "To scale the rows of A there have to be as "
                                 "many rows as given weights. Rows count: "
                              << rows << ", given weights: " << weight.size;
  KernelSize kernelSize = getKernelSize(rows * columns);
  weightRows<<<kernelSize.gridDim, kernelSize.blockDim>>>(A, rows, columns,
                                                          weight);
}

struct save_inv_sqrt_op {
  __device__ float operator()(float x) const {
    return 1.f / (std::sqrt(x) + 1e-6f);  // epsilon
  }
};
struct sqrt_op {
  __device__ float operator()(float x) const {
    return std::sqrt(x);  // epsilon
  }
};

GPUFloatData JacobiPreconditionedSolver(
    CUBLASWrapper& cublas, CUBWrapper& cub, CUSOLVERWrapper& cusolver,
    SolverType solver, const GPUFloatData& A, int64_t A_n_rows,
    int64_t A_n_columns, const GPUFloatData& b, const size_t max_iter,
    const float tol) {
  // col_norms_2 = sum(A**2, axis=0) + epsilon
  // Define boundaries of columns in A
  GPUIntData column_offsets{static_cast<size_t>(A_n_columns + 1)};
  thrust::sequence(thrust::device, column_offsets.data.get(),
                   column_offsets.data.get() + column_offsets.size, 0,
                   static_cast<int>(A_n_rows));
  // Compute squared sum of columns
  GPUFloatData col_norms_2 = cub.deviceSegmentedReduceSquaredSum<float>(
      A, A_n_columns, column_offsets, column_offsets, 1);
  // D = 1.f / sqrt(col_norms_2)
  GPUFloatData D{static_cast<size_t>(A_n_columns)};
  thrust::transform(thrust::device, col_norms_2.data.get(),
                    col_norms_2.data.get() + col_norms_2.size, D.data.get(),
                    save_inv_sqrt_op{});
  // A_scaled = A * diag(D)
  GPUFloatData A_scaled =
      cublas.dgmm(A_n_rows, A_n_columns, A, D,
                  true);  // TODO A is not really needed anymore after this, we
                          // could do the scaling inplace

  GPUFloatData x_scaled;
  switch (solver) {
    case SolverType::CGLS:
      x_scaled =
          CGLS(cublas, A_scaled, A_n_rows, A_n_columns, b, max_iter, tol);
      break;
    case SolverType::LSQR:
      x_scaled =
          LSQR(cublas, A_scaled, A_n_rows, A_n_columns, b, max_iter, tol);
      break;
    case SolverType::QR:
      x_scaled = directQR(cusolver, A, A_n_rows, A_n_columns, b);
      break;
    case SolverType::LSMR:
      x_scaled =
          LSMR(cublas, A_scaled, A_n_rows, A_n_columns, b, 0.f, max_iter, tol);
  }
  // Undo scaling to recover solution
  // x = D * x_scaled
  thrust::transform(thrust::device, x_scaled.data.get(),
                    x_scaled.data.get() + x_scaled.size, D.data.get(),
                    x_scaled.data.get(), thrust::multiplies<float>{});
  return x_scaled;
}

struct atomic_squared_sum_op {
  float* accum;

  __device__ void operator()(const thrust::tuple<float, int>& tuple) {
    float value = thrust::get<0>(tuple);
    int col = thrust::get<1>(tuple);
    atomicAdd(accum + col, value * value);
  }
};

struct sparse_scale_op {
  float* D;
  __device__ float operator()(const thrust::tuple<float, int>& tuple) {
    float value = thrust::get<0>(tuple);
    int col = thrust::get<1>(tuple);
    return value * D[col];
  }
};

GPUFloatData JacobiPreconditionedSolver_sparse(
    CUBLASWrapper& cublas, CUSPARSEWrapper& cusparse, CUBWrapper& cub,
    SolverType solver, const GPUIntData& A_csr_row_offsets,
    const GPUIntData& A_csr_column_idxs, const GPUFloatData& A_csr_values,
    int64_t A_n_rows, int64_t A_n_columns, const GPUFloatData& b,
    const size_t max_iter, const float tol) {
  // col_norms_2 = sum(A**2, axis=0) + epsilon
  GPUFloatData col_norms_2{static_cast<size_t>(A_n_columns)};
  thrust::fill(thrust::device, col_norms_2.data.get(),
               col_norms_2.data.get() + col_norms_2.size, 0.f);
  // use atomic add to add up all squared values per column
  // iterator over value-column-idx pairs
  auto values_iter = thrust::make_zip_iterator(A_csr_values.data.get(),
                                               A_csr_column_idxs.data.get());
  thrust::for_each_n(thrust::device, values_iter, A_csr_values.size,
                     atomic_squared_sum_op{col_norms_2.data.get()});
  // D = 1.f / sqrt(col_norms_2)
  GPUFloatData D{static_cast<size_t>(A_n_columns)};
  thrust::transform(thrust::device, col_norms_2.data.get(),
                    col_norms_2.data.get() + col_norms_2.size, D.data.get(),
                    save_inv_sqrt_op{});
  // A_scaled = A * diag(D)
  GPUFloatData A_csr_values_scaled{A_csr_values.size};
  thrust::transform(thrust::device,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        A_csr_values.data.get(), A_csr_column_idxs.data.get())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        A_csr_values.data.get() + A_csr_values.size,
                        A_csr_column_idxs.data.get() + A_csr_column_idxs.size)),
                    A_csr_values_scaled.data.get(),
                    sparse_scale_op{D.data.get()});

  GPUFloatData x_scaled;
  switch (solver) {
    case SolverType::CGLS:
      x_scaled = CGLS_sparse(cublas, cusparse, A_csr_row_offsets,
                             A_csr_column_idxs, A_csr_values_scaled, A_n_rows,
                             A_n_columns, b, max_iter, tol);
      break;
    case SolverType::LSQR:
      x_scaled = LSQR_sparse(cublas, cusparse, A_csr_row_offsets,
                             A_csr_column_idxs, A_csr_values_scaled, A_n_rows,
                             A_n_columns, b, max_iter, tol);
      break;
    case SolverType::QR:
      LOG(FATAL) << "directQR is not implemented for sparse matrices.";
      break;
    case SolverType::LSMR:
      x_scaled = LSMR_sparse(cublas, cusparse, A_csr_row_offsets,
                             A_csr_column_idxs, A_csr_values_scaled, A_n_rows,
                             A_n_columns, b, 0.f, max_iter, tol);
  }
  // Undo scaling to recover solution
  // x = D * x_scaled
  thrust::transform(thrust::device, x_scaled.data.get(),
                    x_scaled.data.get() + x_scaled.size, D.data.get(),
                    x_scaled.data.get(), thrust::multiplies<float>{});
  return x_scaled;
}

GPUFloatData CGLS_sparse(CUBLASWrapper& cublas, CUSPARSEWrapper& cusparse,
                         const GPUIntData& A_csr_row_offsets,
                         const GPUIntData& A_csr_column_idxs,
                         const GPUFloatData& A_csr_values, int64_t A_n_rows,
                         int64_t A_n_columns, const GPUFloatData& b,
                         const size_t max_iter, const float tol) {
  // Verify valid shapes
  CHECK_EQ(A_n_rows, b.size)
      << "In Ax = b A_n_rows (" << A_n_rows << ") and b.size (" << b.size
      << ") have to be the same";

  VLOG(2) << "Runnign sparse CGLS with " << A_n_rows << " knowns and "
          << A_n_columns << " unknowns.";
  // Initialize x with zeros
  auto t1 = std::chrono::high_resolution_clock::now();
  GPUFloatData x(A_n_columns);
  thrust::fill(thrust::device, x.data.get(), x.data.get() + x.size,
               0.f);   // [cols]
  GPUFloatData r = b;  // residual [rows]
  // Allocate gpu arrays for cublas results
  GPUFloatData s(A_n_columns);
  GPUFloatData q(A_n_rows);

  // Prepare sparse multiplications s = A.T @ r
  auto s_is_ATr = cusparse.spmv_prepare(
      A_n_rows, A_n_columns, A_csr_row_offsets, A_csr_column_idxs, A_csr_values,
      r, s, 1.f, 0.f, true, true);
  // compute gradient s = A.T @ r
  cusparse.spmv(s_is_ATr, 1.f, 0.f, true);
  GPUFloatData p = s;
  float gamma_old = cublas.dot(s, s);  // squared norm

  // Prepare sparse multiplications q = A @ p
  auto q_is_Ap = cusparse.spmv_prepare(A_n_rows, A_n_columns, A_csr_row_offsets,
                                       A_csr_column_idxs, A_csr_values, p, q);

  bool has_stopped_early = false;
  for (size_t k = 0; k < max_iter; ++k) {
    float gradient_norm = std::sqrt(gamma_old);
    VLOG(5) << "CGLS iter " << k << ": gradient_norm = " << gradient_norm;
    // Check if gradient norm is smaller than the tolerance
    if (gradient_norm < tol) {
      VLOG(3) << "Sparse_CGLS stopping criterion reached in iteraion " << k;
      VLOG(3) << "Stopping criterion is gradient_norm < tol";
      VLOG(3) << gradient_norm << " < " << tol;
      has_stopped_early = true;
      break;
    }
    // compute q = A @ p
    cusparse.spmv(q_is_Ap);
    // compute step size
    float alpha = gamma_old / (cublas.dot(q, q));
    // update solution and residual
    cublas.axpy(alpha, p, x);   // x += alpha * p
    cublas.axpy(-alpha, q, r);  // r -= alpha * q
    // Compute new gradient s = A.T @ r
    cusparse.spmv(s_is_ATr, 1.f, 0.f, true);
    float gamma_new = cublas.dot(s, s);
    // Update search direction
    float beta = gamma_new / gamma_old;
    cublas.scale(p, beta);
    cublas.axpy(1.f, s, p);  // p = s + beta * p

    gamma_old = gamma_new;
  }
  if (!has_stopped_early) {
    VLOG(3) << "Sparse_CGLS stopped after max_iterations = " << max_iter;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto passed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

  VLOG(2) << "Sparse_CGLS done after " << passed_time;
  return x;
}

GPUFloatData CGLS(CUBLASWrapper& cublas, const GPUFloatData& A,
                  int64_t A_n_rows, int64_t A_n_columns, const GPUFloatData& b,
                  const size_t max_iter, const float tol) {
  // Verify valid shapes
  CHECK_EQ(A_n_rows * A_n_columns, A.size)
      << "A_n_rows * A_n_columns  (" << A_n_rows * A_n_columns
      << ") have to be equal to A.size (" << A.size << ").";
  CHECK_EQ(A_n_rows, b.size)
      << "In Ax = b A_n_rows (" << A_n_rows << ") and b.size (" << b.size
      << ") have to be the same";

  // Initialize x with zeros
  VLOG(2) << "Runnign CGLS with " << A_n_rows << " knowns and " << A_n_columns
          << " unknowns.";
  auto t1 = std::chrono::high_resolution_clock::now();
  GPUFloatData x(A_n_columns);
  thrust::fill(thrust::device, x.data.get(), x.data.get() + x.size,
               0.f);   // [cols]
  GPUFloatData r = b;  // residual [rows]
  GPUFloatData s = cublas.gemv(A_n_rows, A_n_columns, A, r, true,
                               true);  // gradient A.T @ r [cols]
  GPUFloatData p = s;                  // search direction
  float gamma_old = cublas.dot(s, s);  // squared norm

  // preallocate gpu array for q
  GPUFloatData q(A_n_rows);

  bool has_stopped_early = false;
  for (size_t k = 0; k < max_iter; ++k) {
    float gradient_norm = std::sqrt(gamma_old);
    VLOG(5) << "CGLS iter " << k << ": gradient_norm = " << gradient_norm;
    // Check if gradient norm is smaller than the tolerance
    if (gradient_norm < tol) {
      VLOG(3) << "CGLS stopping criterion reached in iteraion " << k;
      VLOG(3) << "Stopping criterion is gradient_norm < tol";
      VLOG(3) << gradient_norm << " < " << tol;
      has_stopped_early = true;
      break;
    }
    // q = A @ p
    cublas.gemv(A_n_rows, A_n_columns, A, p, q, false, true);
    // compute step size
    float alpha = gamma_old / cublas.dot(q, q);
    // update solution and residual
    cublas.axpy(alpha, p, x);   // x += alpha * p
    cublas.axpy(-alpha, q, r);  // r -= alpha * q
    // Compute new gradient s = A.T @ r
    cublas.gemv(A_n_rows, A_n_columns, A, r, s, true, true);
    float gamma_new = cublas.dot(s, s);
    // Update search direction
    float beta = gamma_new / gamma_old;
    cublas.scale(p, beta);
    cublas.axpy(1.f, s, p);  // p = s + beta * p

    gamma_old = gamma_new;
  }
  if (!has_stopped_early) {
    VLOG(3) << "CGLS stopped after max_iterations = " << max_iter;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto passed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  VLOG(2) << "CGLS done after " << passed_time;
  return x;
}

void sym_ortho(float a, float b, float& c, float& s, float& rho) {
  // Numeric stability for rotations
  // See Page 22 of:
  // S.-C. Choi, "Iterative Methods for Singular Linear Equations
  //                  and Least-Squares Problems", Dissertation,
  //         http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
  if (b == 0.f) {
    c = std::copysign(1.f, a);
    s = 0.f;
    rho = std::abs(a);
  } else if (a == 0.f) {
    c = 0.f;
    s = std::copysign(1.f, b);
    rho = std::abs(b);
  } else if (std::abs(b) > std::abs(a)) {
    float tau = a / b;
    s = std::copysign(1.f / std::sqrt(1.f + tau * tau), b);
    c = s * tau;
    rho = b / s;
  } else {
    float tau = b / a;
    c = std::copysign(1.f / std::sqrt(1.f + tau * tau), a);
    s = c * tau;
    rho = a / c;
  }
}

GPUFloatData LSQR(CUBLASWrapper& cublas, const GPUFloatData& A,
                  int64_t A_n_rows, int64_t A_n_columns, const GPUFloatData& b,
                  const size_t max_iter, const float tol, const float CONDLIM) {
  CHECK_EQ(A_n_rows * A_n_columns, A.size)
      << "A_n_rows * A_n_columns  (" << A_n_rows * A_n_columns
      << ") have to be equal to A.size (" << A.size << ").";
  CHECK_EQ(A_n_rows, b.size)
      << "In Ax = b A_n_rows (" << A_n_rows << ") and b.size (" << b.size
      << ") have to be the same";

  // Initialize x with zeros
  VLOG(2) << "Runnign LSQR with " << A_n_rows << " knowns and " << A_n_columns
          << " unknowns.";
  auto t1 = std::chrono::high_resolution_clock::now();
  // x = zeros(columns)
  GPUFloatData x(A_n_columns);
  thrust::fill(thrust::device, x.data.get(), x.data.get() + x.size, 0.f);
  float beta = cublas.norm2(b);
  // u = b / beta
  GPUFloatData u = b;
  cublas.scale(u, 1.f / beta);
  // v = A.T @ u
  GPUFloatData v = cublas.gemv(A_n_rows, A_n_columns, A, u, true, true);
  // alpha = norm(v)
  float alpha = cublas.norm2(v);
  float alpha_prev = alpha;
  // v /= alpha
  cublas.scale(v, 1.f / alpha);
  // w = v.copy
  GPUFloatData w = v;
  float phi_bar = beta;
  float rho_bar = alpha;

  // diagonal matrix storage
  float theta = 0.f;

  // Norms for stopping criterion (are updated on the fly)
  float b_norm = cublas.norm2(b);
  float A_norm_2 = 0.f;
  float D_norm_2 = 0.f;
  float w_norm_2 = 0.f;
  float cond_A_estimate = 1.f;
  float x_norm = 0.f;

  // Memory allocation for matrix multiplications
  GPUFloatData Av{static_cast<size_t>(A_n_rows)};
  GPUFloatData ATu{static_cast<size_t>(A_n_columns)};

  bool has_stopped_early = false;
  for (size_t iter = 0; iter < max_iter; ++iter) {
    // Bigiagonalization step
    // u = A @ v - alpha * u
    cublas.scale(u, -alpha);  // u = -alpha * u
    cublas.gemv(A_n_rows, A_n_columns, A, v, Av, false, true);  // A@v
    cublas.axpy(1.f, Av, u);
    // beta = norm(u)
    beta = cublas.norm2(u);
    // u /= beta
    cublas.scale(u, 1.f / beta);

    // v = A.T @ u - beta * v
    cublas.scale(v, -beta);
    cublas.gemv(A_n_rows, A_n_columns, A, u, ATu, true, true);  // A.T@u
    cublas.axpy(1.f, ATu, v);
    // alpha = norm(v)
    alpha = cublas.norm2(v);
    // v /= alpha
    cublas.scale(v, 1.f / alpha);

    // Update rotation parameters
    float s_new, c_new, rho;
    sym_ortho(rho_bar, beta, c_new, s_new, rho);
    theta = s_new * alpha;
    rho_bar = -c_new * alpha;
    float phi = c_new * phi_bar;
    phi_bar = s_new * phi_bar;

    // Update solution
    // x += (phi /rho) * w
    float phi_over_rho = phi / rho;
    cublas.axpy(phi_over_rho, w, x);
    // w = v - (theta / rho) * w
    cublas.scale(w, -(theta / rho));  // w = -(theta / rho) * w
    cublas.axpy(1.f, v, w);

    // Update norm estimates for stopping criterion
    float residual_norm = phi_bar;
    float ATr_norm = phi_bar * alpha * std::abs(c_new);
    x_norm = std::sqrt(x_norm * x_norm + phi_over_rho * phi_over_rho);
    A_norm_2 += alpha_prev * alpha_prev + beta * beta;
    alpha_prev = alpha;
    float new_w_norm_2 = (1.f + theta * theta * w_norm_2) / (rho * rho);
    D_norm_2 += new_w_norm_2;
    w_norm_2 = new_w_norm_2;
    float A_norm = std::sqrt(A_norm_2);
    float D_norm = std::sqrt(D_norm_2);
    cond_A_estimate = A_norm * D_norm;

    float crit_1_term = tol * b_norm + tol * A_norm * x_norm;
    bool stopping_criterion_1 = residual_norm <= crit_1_term;
    float crit_2_term = ATr_norm / (A_norm * residual_norm);
    VLOG(5) << "LSQR iter " << iter << ": residual norm = " << residual_norm
            << ", ||ATr||/(||A||*||r||) = " << crit_2_term
            << ", cond(A) = " << cond_A_estimate;
    bool stopping_criterion_2 = crit_2_term <= tol;
    bool stopping_criterion_3 = cond_A_estimate >= CONDLIM;
    if (stopping_criterion_1) {
      VLOG(3) << "LSQR stopping criterion 1 reached in iteraion " << iter;
      VLOG(3) << "Stopping criterion is ||r|| <= tol*||b||+tol*||A||*||x||";
      VLOG(3) << residual_norm << " <= " << crit_1_term;
      has_stopped_early = true;
      break;
    }
    if (stopping_criterion_2) {
      VLOG(3) << "LSQR stopping criterion 2 reached in iteraion " << iter;
      VLOG(3) << "Stopping criterion is ||A.T*r||/(||A||*||r||) <= tol";
      VLOG(3) << crit_2_term << " <= " << tol;
      has_stopped_early = true;
      break;
    }
    if (stopping_criterion_3) {
      VLOG(3) << "LSQR stopping criterion 3 reached in iteraion " << iter;
      VLOG(3) << "Stopping criterion is cond(A) >= CONDLIM";
      VLOG(3) << cond_A_estimate << " >= " << CONDLIM;
      has_stopped_early = true;
      break;
    }
  }
  if (!has_stopped_early) {
    VLOG(3) << "LSQR stopped after max_iterations = " << max_iter;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto passed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  VLOG(2) << "LSQR done after " << passed_time;
  return x;
}

GPUFloatData LSQR_sparse(CUBLASWrapper& cublas, CUSPARSEWrapper& cusparse,
                         const GPUIntData& A_csr_row_offsets,
                         const GPUIntData& A_csr_column_idxs,
                         const GPUFloatData& A_csr_values, int64_t A_n_rows,
                         int64_t A_n_columns, const GPUFloatData& b,
                         const size_t max_iter, const float tol,
                         const float CONDLIM) {
  CHECK_EQ(A_n_rows, b.size)
      << "In Ax = b A_n_rows (" << A_n_rows << ") and b.size (" << b.size
      << ") have to be the same";

  // Initialize x with zeros
  VLOG(2) << "Runnign sparse_LSQR with " << A_n_rows << " knowns and "
          << A_n_columns << " unknowns.";
  auto t1 = std::chrono::high_resolution_clock::now();
  // x = zeros(columns)
  GPUFloatData x(A_n_columns);
  thrust::fill(thrust::device, x.data.get(), x.data.get() + x.size, 0.f);
  float beta = cublas.norm2(b);
  // u = b / beta
  GPUFloatData u = b;
  cublas.scale(u, 1.f / beta);

  // Prepare ATu = A.T @ u multiplication
  GPUFloatData ATu{static_cast<size_t>(A_n_columns)};
  CusparseSpMVHandles ATu_multiplication = cusparse.spmv_prepare(
      A_n_rows, A_n_columns, A_csr_row_offsets, A_csr_column_idxs, A_csr_values,
      u, ATu, 1.f, 0.f, true, true);
  // run ATu = A.T@u
  cusparse.spmv(ATu_multiplication, 1.f, 0.f, true);
  // v = A.T @ u
  GPUFloatData v = ATu;

  // alpha = norm(v)
  float alpha = cublas.norm2(v);
  float alpha_prev = alpha;
  // v /= alpha
  cublas.scale(v, 1.f / alpha);
  // w = v.copy
  GPUFloatData w = v;
  float phi_bar = beta;
  float rho_bar = alpha;

  // diagonal matrix storage
  float theta = 0.f;

  // Norms for stopping criterion (are updated on the fly)
  float b_norm = cublas.norm2(b);
  float A_norm_2 = 0.f;
  float D_norm_2 = 0.f;
  float w_norm_2 = 0.f;
  float cond_A_estimate = 1.f;
  float x_norm = 0.f;

  // Prepare Av = A @ v
  GPUFloatData Av{static_cast<size_t>(A_n_rows)};
  CusparseSpMVHandles Av_multiplication =
      cusparse.spmv_prepare(A_n_rows, A_n_columns, A_csr_row_offsets,
                            A_csr_column_idxs, A_csr_values, v, Av);

  bool has_stopped_early = false;
  for (size_t iter = 0; iter < max_iter; ++iter) {
    // Bigiagonalization step
    // u = A @ v - alpha * u
    cublas.scale(u, -alpha);  // u = -alpha * u
    // run Av = A@v
    cusparse.spmv(Av_multiplication);
    cublas.axpy(1.f, Av, u);
    // beta = norm(u)
    beta = cublas.norm2(u);
    // u /= beta
    cublas.scale(u, 1.f / beta);

    // v = A.T @ u - beta * v
    cublas.scale(v, -beta);
    // run ATu = A.T@u
    cusparse.spmv(ATu_multiplication, 1.f, 0.f, true);
    cublas.axpy(1.f, ATu, v);
    // alpha = norm(v)
    alpha = cublas.norm2(v);
    // v /= alpha
    cublas.scale(v, 1.f / alpha);

    // Update rotation parameters
    float s_new, c_new, rho;
    sym_ortho(rho_bar, beta, c_new, s_new, rho);
    theta = s_new * alpha;
    rho_bar = -c_new * alpha;
    float phi = c_new * phi_bar;
    phi_bar = s_new * phi_bar;

    // Update solution
    // x += (phi /rho) * w
    float phi_over_rho = phi / rho;
    cublas.axpy(phi_over_rho, w, x);
    // w = v - (theta / rho) * w
    cublas.scale(w, -(theta / rho));  // w = -(theta / rho) * w
    cublas.axpy(1.f, v, w);

    // Update norm estimates for stopping criterion
    float residual_norm = phi_bar;
    float ATr_norm = phi_bar * alpha * std::abs(c_new);
    x_norm = std::sqrt(x_norm * x_norm + phi_over_rho * phi_over_rho);
    A_norm_2 += alpha_prev * alpha_prev + beta * beta;
    alpha_prev = alpha;
    float new_w_norm_2 = (1.f + theta * theta * w_norm_2) / (rho * rho);
    D_norm_2 += new_w_norm_2;
    w_norm_2 = new_w_norm_2;
    float A_norm = std::sqrt(A_norm_2);
    float D_norm = std::sqrt(D_norm_2);
    cond_A_estimate = A_norm * D_norm;

    float crit_1_term = tol * b_norm + tol * A_norm * x_norm;
    bool stopping_criterion_1 = residual_norm <= crit_1_term;
    float crit_2_term = ATr_norm / (A_norm * residual_norm);
    VLOG(5) << "sparse_LSQR iter " << iter << ": ||r|| = " << residual_norm
            << ", ||ATr||/(||A||*||r||) = " << crit_2_term
            << ", cond(A) = " << cond_A_estimate;
    bool stopping_criterion_2 = crit_2_term <= tol;
    bool stopping_criterion_3 = cond_A_estimate >= CONDLIM;
    if (stopping_criterion_1) {
      VLOG(3) << "sparse_LSQR stopping criterion 1 reached in iteraion "
              << iter;
      VLOG(3) << "Stopping criterion is ||r|| <= tol*||b||+tol*||A||*||x||";
      VLOG(3) << residual_norm << " <= " << crit_1_term;
      has_stopped_early = true;
      break;
    }
    if (stopping_criterion_2) {
      VLOG(3) << "sparse_LSQR stopping criterion 2 reached in iteraion "
              << iter;
      VLOG(3) << "Stopping criterion is ||A.T*r||/(||A||*||r||) <= tol";
      VLOG(3) << crit_2_term << " <= " << tol;
      has_stopped_early = true;
      break;
    }
    if (stopping_criterion_3) {
      VLOG(3) << "LSQR stopping criterion 3 reached in iteraion " << iter;
      VLOG(3) << "Stopping criterion is cond(A) >= CONDLIM";
      VLOG(3) << cond_A_estimate << " >= " << CONDLIM;
      has_stopped_early = true;
      break;
    }
  }
  if (!has_stopped_early) {
    VLOG(3) << "sparse_LSQR stopped after max_iterations = " << max_iter;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto passed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  VLOG(2) << "sparse_LSQR done after " << passed_time;
  return x;
}

GPUFloatData LSMR(CUBLASWrapper& cublas, const GPUFloatData& A,
                  int64_t A_n_rows, int64_t A_n_columns, const GPUFloatData& b,
                  float dampening, const size_t max_iter, const float tol,
                  const float CONDLIM) {
  // See https://stanford.edu/group/SOL/software/lsmr/LSMR-SISC-2011.pdf
  CHECK_EQ(A_n_rows * A_n_columns, A.size)
      << "A_n_rows * A_n_columns  (" << A_n_rows * A_n_columns
      << ") have to be equal to A.size (" << A.size << ").";
  CHECK_EQ(A_n_rows, b.size)
      << "In Ax = b A_n_rows (" << A_n_rows << ") and b.size (" << b.size
      << ") have to be the same";

  auto t1 = std::chrono::high_resolution_clock::now();
  // # Step 1: Initialize
  //   beta = norm(b)
  float beta = cublas.norm2(b);
  float b_norm = beta;
  //   u = b / beta_1
  GPUFloatData u = b;
  cublas.scale(u, 1.f / beta);
  //   v = transpose(A) @ u
  GPUFloatData v = cublas.gemv(A_n_rows, A_n_columns, A, u, true, true);
  //   alpha = norm(v)
  float alpha = cublas.norm2(v);
  //   v = v / alpha_1
  cublas.scale(v, 1.f / alpha);
  //
  //   # Recurrence variables (Sec 2.8)
  float alpha_bar = alpha;
  float zeta_bar = alpha * beta;
  float rho_prev = 1.f;
  float rho_bar_prev = 1.f;
  float c_bar_prev = 1.f;
  float s_bar_prev = 0.f;
  float beta_dot_dot = beta;
  float beta_dot = 0.f;
  float rho_dot_prev = 1.f;
  float tau_tilde_old = 0.f;
  float theta_tilde_prev = 0.f;
  float zeta_prev = 0.f;
  float d = 0.f;
  GPUFloatData h = v;
  GPUFloatData h_bar{h.size};
  thrust::fill(thrust::device, h_bar.data.get(), h_bar.data.get() + h_bar.size,
               0.f);
  GPUFloatData x(A_n_columns);
  thrust::fill(thrust::device, x.data.get(), x.data.get() + x.size, 0.f);

  // For iterative norm estimation
  float A_norm_sq = 0.f;
  float x_norm_sq = 0.f;
  float max_rho_bar = 0.f;
  float min_rho_bar = std::numeric_limits<float>::infinity();
  // Memory allocation for matrix multiplications
  GPUFloatData Av{static_cast<size_t>(A_n_rows)};
  GPUFloatData ATu{static_cast<size_t>(A_n_columns)};
  //   for k in 1 to max_iter:
  bool has_stopped_early = false;
  for (size_t iter = 0; iter < max_iter; ++iter) {
    //       # Step 3: Bidiagonalization
    //       u = A @ v - alpha_prev * u
    cublas.scale(u, -alpha);  // u = -alpha * u
    cublas.gemv(A_n_rows, A_n_columns, A, v, Av, false, true);  // A@v
    cublas.axpy(1.f, Av, u);
    //       beta = norm(u)
    beta = cublas.norm2(u);
    //       u = u / beta
    cublas.scale(u, 1.f / beta);
    //       v = transpose(A) @ u - beta * v
    cublas.scale(v, -beta);
    cublas.gemv(A_n_rows, A_n_columns, A, u, ATu, true, true);  // A.T@u
    cublas.axpy(1.f, ATu, v);
    //       alpha = norm(v)
    float alpha_prev = alpha;
    alpha = cublas.norm2(v);
    //       v = v / alpha
    cublas.scale(v, 1.f / alpha);
    //
    //       # Step 4: Rotation P_hat_k
    float alpha_hat, c_hat, s_hat;
    sym_ortho(alpha_bar, dampening, c_hat, s_hat, alpha_hat);

    //       # Step 5: Rotation P_k
    float rho, c, s;
    sym_ortho(alpha_hat, beta, c, s, rho);
    //       theta = s * alpha
    float theta_next = s * alpha;
    //       alpha_bar_next = c * alpha
    float alpha_bar_next = c * alpha;
    //
    //       # Step 6: Rotation P_bar_k
    //       theta_bar = s_bar_prev * rho
    float theta_bar = s_bar_prev * rho;
    //       rho_bar = sqrt((c_bar_prev * rho)**2 + theta**2)
    //       c_bar = c_bar_prev * rho / rho_tilde
    //       s_bar = theta / rho_bar
    float rho_bar, c_bar, s_bar;
    sym_ortho(c_bar_prev * rho, theta_next, c_bar, s_bar, rho_bar);
    //       zeta = c_bar * zeta_bar
    float zeta = c_bar * zeta_bar;
    //       zeta_bar_next = -s_bar * zeta_bar
    float zeta_bar_next = -s_bar * zeta_bar;

    //       # Step 7: Update h, h_bar, x
    //       h_bar_new = h - (theta_bar * rho / (rho_prev * rho_bar_prev)) *
    //       h_bar
    cublas.scale(h_bar, -(theta_bar * rho / (rho_prev * rho_bar_prev)));
    cublas.axpy(1.f, h, h_bar);
    //       x += (zeta / (rho * rho_bar)) * h_bar_new
    cublas.axpy((zeta / (rho * rho_bar)), h_bar, x);
    //       h_next = v - (theta/ rho) * h
    cublas.scale(h, -(theta_next / rho));
    cublas.axpy(1.f, v, h);
    //
    //       # Sep 8 Apply rotation P_hat and P_k
    float beta_tick = c_hat * beta_dot_dot;
    float beta_vee = -s_hat * beta_dot_dot;
    float beta_hat = c * beta_tick;
    beta_dot_dot = -s * beta_tick;
    //       # Step 9 If k>= 2 construct and apply rotation P_tilde_old
    float rho_tilde_prev, c_tilde_prev, s_tilde_prev;
    sym_ortho(rho_dot_prev, theta_bar, c_tilde_prev, s_tilde_prev,
              rho_tilde_prev);
    float theta_tilde = s_tilde_prev * rho_bar;
    float rho_dot = c_tilde_prev * rho_bar;
    float beta_tilde_prev = c_tilde_prev * beta_dot + s_tilde_prev * beta_hat;
    beta_dot = -s_tilde_prev * beta_dot + c_tilde_prev * beta_hat;
    //       # Step 10 update t_tilde by forward substitution
    tau_tilde_old =
        (zeta_prev - theta_tilde_prev * tau_tilde_old) / rho_tilde_prev;
    float tau_dot = (zeta - theta_tilde * tau_tilde_old) / rho_dot;

    //       # Step 11 Compute ||r_k||
    d += beta_vee * beta_vee;
    float gamma = d + (beta_dot - tau_dot) * (beta_dot - tau_dot) +
                  beta_dot_dot * beta_dot_dot;
    float r_norm = std::sqrt(gamma);

    float ATr_norm = abs(zeta_bar_next);
    //       $ Step 12 Compute ||A.Tr||, ||A|| and cond(A)
    A_norm_sq += alpha_prev * alpha_prev + beta * beta;
    //       x_norm_sq += (zeta / (rho * rho_bar))**2
    x_norm_sq += (zeta / (rho * rho_bar)) * (zeta / (rho * rho_bar));
    float x_norm = std::sqrt(x_norm_sq);
    //       cond_estimate = max_rho_bar / min_rho_bar
    if (iter > 0) {
      min_rho_bar = std::min(min_rho_bar, rho_bar_prev);
    }
    max_rho_bar = std::max(max_rho_bar, rho_bar_prev);
    float cond_estimate = max_rho_bar / min_rho_bar;
    //       S1 = r_norm <= tol * (norm(b) + A_norm*x_norm)
    float crit_1_term = tol * (b_norm + std::sqrt(A_norm_sq) * x_norm);
    bool stopping_criterion_1 = r_norm <= crit_1_term;
    //       S2 = ATr_norm <= tol * sqrt(A_norm_sq) * r_norm
    float crit_2_term = ATr_norm / (std::sqrt(A_norm_sq) * r_norm);
    bool stopping_criterion_2 = crit_2_term <= tol;
    //       S3 = cond_estimate >= CONDLIM
    bool stopping_criterion_3 = cond_estimate >= CONDLIM;
    VLOG(5) << "LSMR iter " << iter << ": ||r|| = " << r_norm
            << ", ||ATr||/(||A||*||r||) = " << crit_2_term
            << ", cond(A) = " << cond_estimate;

    //       if S1 or S2 or S3:
    //           break
    if (stopping_criterion_1) {
      VLOG(3) << "LSMR stopping criterion 1 reached in iteraion " << iter;
      VLOG(3) << "Stopping criterion is ||r|| <= tol*||b||+tol*||A||*||x||";
      VLOG(3) << r_norm << " <= " << crit_1_term;
      has_stopped_early = true;
      break;
    }
    if (stopping_criterion_2) {
      VLOG(3) << "LSMR stopping criterion 2 reached in iteraion " << iter;
      VLOG(3) << "Stopping criterion is ||A.T*r||/(||A||*||r||) <= tol";
      VLOG(3) << crit_2_term << " <= " << tol;
      has_stopped_early = true;
      break;
    }
    if (stopping_criterion_3) {
      VLOG(3) << "LSMR stopping criterion 3 reached in iteraion " << iter;
      VLOG(3) << "Stopping criterion is cond(A) >= CONDLIM";
      VLOG(3) << cond_estimate << " >= " << CONDLIM;
      has_stopped_early = true;
      break;
    }
    //
    //       # Cycle variables for next iteration
    rho_prev = rho;
    alpha_bar = alpha_bar_next;
    rho_bar_prev = rho_bar;
    c_bar_prev = c_bar;
    s_bar_prev = s_bar;
    zeta_bar = zeta_bar_next;
    zeta_prev = zeta;
    theta_tilde_prev = theta_tilde;
    rho_dot_prev = rho_dot;
  }
  if (!has_stopped_early) {
    VLOG(3) << "LSMR stopped after max_iterations = " << max_iter;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto passed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  VLOG(2) << "LSMR done after " << passed_time;
  //   return x
  return x;
}

GPUFloatData LSMR_sparse(CUBLASWrapper& cublas, CUSPARSEWrapper& cusparse,
                         const GPUIntData& A_csr_row_offsets,
                         const GPUIntData& A_csr_column_idxs,
                         const GPUFloatData& A_csr_values, int64_t A_n_rows,
                         int64_t A_n_columns, const GPUFloatData& b,
                         float dampening, const size_t max_iter,
                         const float tol, const float CONDLIM) {
  // See https://stanford.edu/group/SOL/software/lsmr/LSMR-SISC-2011.pdf
  CHECK_EQ(A_n_rows, b.size)
      << "In Ax = b A_n_rows (" << A_n_rows << ") and b.size (" << b.size
      << ") have to be the same";

  auto t1 = std::chrono::high_resolution_clock::now();
  // # Step 1: Initialize
  //   beta = norm(b)
  float beta = cublas.norm2(b);
  float b_norm = beta;
  //   u = b / beta_1
  GPUFloatData u = b;
  cublas.scale(u, 1.f / beta);
  //   v = transpose(A) @ u
  // Prepare ATu = A.T@u multiplication
  GPUFloatData ATu{static_cast<size_t>(A_n_columns)};
  CusparseSpMVHandles ATu_handle = cusparse.spmv_prepare(
      A_n_rows, A_n_columns, A_csr_row_offsets, A_csr_column_idxs, A_csr_values,
      u, ATu, 1.f, 0.f, true, true);
  cusparse.spmv(ATu_handle, 1.f, 0.f, true);  // A.T@u
  GPUFloatData v = ATu;
  //   alpha = norm(v)
  float alpha = cublas.norm2(v);
  //   v = v / alpha_1
  cublas.scale(v, 1.f / alpha);

  // Prepare Av = A@v multiplication
  GPUFloatData Av{static_cast<size_t>(A_n_rows)};
  CusparseSpMVHandles Av_handle = cusparse.spmv_prepare(
      A_n_rows, A_n_columns, A_csr_row_offsets, A_csr_column_idxs, A_csr_values,
      v, Av, 1.f, 0.f, true, false);

  //   # Recurrence variables (Sec 2.8)
  float alpha_bar = alpha;
  float zeta_bar = alpha * beta;
  float rho_prev = 1.f;
  float rho_bar_prev = 1.f;
  float c_bar_prev = 1.f;
  float s_bar_prev = 0.f;
  float beta_dot_dot = beta;
  float beta_dot = 0.f;
  float rho_dot_prev = 1.f;
  float tau_tilde_old = 0.f;
  float theta_tilde_prev = 0.f;
  float zeta_prev = 0.f;
  float d = 0.f;
  GPUFloatData h = v;
  GPUFloatData h_bar{h.size};
  thrust::fill(thrust::device, h_bar.data.get(), h_bar.data.get() + h_bar.size,
               0.f);
  GPUFloatData x(A_n_columns);
  thrust::fill(thrust::device, x.data.get(), x.data.get() + x.size, 0.f);

  // For iterative norm estimation
  float A_norm_sq = 0.f;
  float x_norm_sq = 0.f;
  float max_rho_bar = 0.f;
  float min_rho_bar = std::numeric_limits<float>::infinity();
  //   for k in 1 to max_iter:
  bool has_stopped_early = false;
  for (size_t iter = 0; iter < max_iter; ++iter) {
    //       # Step 3: Bidiagonalization
    //       u = A @ v - alpha_prev * u
    cublas.scale(u, -alpha);                    // u = -alpha * u
    cusparse.spmv(Av_handle, 1.f, 0.f, false);  // A@v
    cublas.axpy(1.f, Av, u);
    //       beta = norm(u)
    beta = cublas.norm2(u);
    //       u = u / beta
    cublas.scale(u, 1.f / beta);
    //       v = transpose(A) @ u - beta * v
    cublas.scale(v, -beta);
    cusparse.spmv(ATu_handle, 1.f, 0.f, true);  // A.T@u
    cublas.axpy(1.f, ATu, v);
    //       alpha = norm(v)
    float alpha_prev = alpha;
    alpha = cublas.norm2(v);
    //       v = v / alpha
    cublas.scale(v, 1.f / alpha);
    //
    //       # Step 4: Rotation P_hat_k
    float alpha_hat, c_hat, s_hat;
    sym_ortho(alpha_bar, dampening, c_hat, s_hat, alpha_hat);

    //       # Step 5: Rotation P_k
    float rho, c, s;
    sym_ortho(alpha_hat, beta, c, s, rho);
    //       theta = s * alpha
    float theta_next = s * alpha;
    //       alpha_bar_next = c * alpha
    float alpha_bar_next = c * alpha;
    //
    //       # Step 6: Rotation P_bar_k
    //       theta_bar = s_bar_prev * rho
    float theta_bar = s_bar_prev * rho;
    //       rho_bar = sqrt((c_bar_prev * rho)**2 + theta**2)
    //       c_bar = c_bar_prev * rho / rho_tilde
    //       s_bar = theta / rho_bar
    float rho_bar, c_bar, s_bar;
    sym_ortho(c_bar_prev * rho, theta_next, c_bar, s_bar, rho_bar);
    //       zeta = c_bar * zeta_bar
    float zeta = c_bar * zeta_bar;
    //       zeta_bar_next = -s_bar * zeta_bar
    float zeta_bar_next = -s_bar * zeta_bar;

    //       # Step 7: Update h, h_bar, x
    //       h_bar_new = h - (theta_bar * rho / (rho_prev * rho_bar_prev)) *
    //       h_bar
    cublas.scale(h_bar, -(theta_bar * rho / (rho_prev * rho_bar_prev)));
    cublas.axpy(1.f, h, h_bar);
    //       x += (zeta / (rho * rho_bar)) * h_bar_new
    cublas.axpy((zeta / (rho * rho_bar)), h_bar, x);
    //       h_next = v - (theta/ rho) * h
    cublas.scale(h, -(theta_next / rho));
    cublas.axpy(1.f, v, h);
    //
    //       # Sep 8 Apply rotation P_hat and P_k
    float beta_tick = c_hat * beta_dot_dot;
    float beta_vee = -s_hat * beta_dot_dot;
    float beta_hat = c * beta_tick;
    beta_dot_dot = -s * beta_tick;
    //       # Step 9 If k>= 2 construct and apply rotation P_tilde_old
    float rho_tilde_prev, c_tilde_prev, s_tilde_prev;
    sym_ortho(rho_dot_prev, theta_bar, c_tilde_prev, s_tilde_prev,
              rho_tilde_prev);
    float theta_tilde = s_tilde_prev * rho_bar;
    float rho_dot = c_tilde_prev * rho_bar;
    float beta_tilde_prev = c_tilde_prev * beta_dot + s_tilde_prev * beta_hat;
    beta_dot = -s_tilde_prev * beta_dot + c_tilde_prev * beta_hat;
    //       # Step 10 update t_tilde by forward substitution
    tau_tilde_old =
        (zeta_prev - theta_tilde_prev * tau_tilde_old) / rho_tilde_prev;
    float tau_dot = (zeta - theta_tilde * tau_tilde_old) / rho_dot;

    //       # Step 11 Compute ||r_k||
    d += beta_vee * beta_vee;
    float gamma = d + (beta_dot - tau_dot) * (beta_dot - tau_dot) +
                  beta_dot_dot * beta_dot_dot;
    float r_norm = std::sqrt(gamma);

    float ATr_norm = abs(zeta_bar_next);
    //       $ Step 12 Compute ||A.Tr||, ||A|| and cond(A)
    A_norm_sq += alpha_prev * alpha_prev + beta * beta;
    //       x_norm_sq += (zeta / (rho * rho_bar))**2
    x_norm_sq += (zeta / (rho * rho_bar)) * (zeta / (rho * rho_bar));
    float x_norm = std::sqrt(x_norm_sq);
    //       cond_estimate = max_rho_bar / min_rho_bar
    if (iter > 0) {
      min_rho_bar = std::min(min_rho_bar, rho_bar_prev);
    }
    max_rho_bar = std::max(max_rho_bar, rho_bar_prev);
    float cond_estimate = max_rho_bar / min_rho_bar;
    //       S1 = r_norm <= tol * (norm(b) + A_norm*x_norm)
    float crit_1_term = tol * (b_norm + std::sqrt(A_norm_sq) * x_norm);
    bool stopping_criterion_1 = r_norm <= crit_1_term;
    //       S2 = ATr_norm <= tol * sqrt(A_norm_sq) * r_norm
    float crit_2_term = ATr_norm / (std::sqrt(A_norm_sq) * r_norm);
    bool stopping_criterion_2 = crit_2_term <= tol;
    //       S3 = cond_estimate >= CONDLIM
    bool stopping_criterion_3 = cond_estimate >= CONDLIM;
    VLOG(5) << "LSMR iter " << iter << ": ||r|| = " << r_norm
            << ", ||ATr||/(||A||*||r||) = " << crit_2_term
            << ", cond(A) = " << cond_estimate;

    //       if S1 or S2 or S3:
    //           break
    if (stopping_criterion_1) {
      VLOG(3) << "LSMR_sparse stopping criterion 1 reached in iteraion "
              << iter;
      VLOG(3) << "Stopping criterion is ||r|| <= tol*||b||+tol*||A||*||x||";
      VLOG(3) << r_norm << " <= " << crit_1_term;
      has_stopped_early = true;
      break;
    }
    if (stopping_criterion_2) {
      VLOG(3) << "LSMR_sparse stopping criterion 2 reached in iteraion "
              << iter;
      VLOG(3) << "Stopping criterion is ||A.T*r||/(||A||*||r||) <= tol";
      VLOG(3) << crit_2_term << " <= " << tol;
      has_stopped_early = true;
      break;
    }
    if (stopping_criterion_3) {
      VLOG(3) << "LSMR_sparse stopping criterion 3 reached in iteraion "
              << iter;
      VLOG(3) << "Stopping criterion is cond(A) >= CONDLIM";
      VLOG(3) << cond_estimate << " >= " << CONDLIM;
      has_stopped_early = true;
      break;
    }
    //
    //       # Cycle variables for next iteration
    rho_prev = rho;
    alpha_bar = alpha_bar_next;
    rho_bar_prev = rho_bar;
    c_bar_prev = c_bar;
    s_bar_prev = s_bar;
    zeta_bar = zeta_bar_next;
    zeta_prev = zeta;
    theta_tilde_prev = theta_tilde;
    rho_dot_prev = rho_dot;
  }
  if (!has_stopped_early) {
    VLOG(3) << "LSMR_sparse stopped after max_iterations = " << max_iter;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto passed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  VLOG(2) << "LSMR_sparse done after " << passed_time;
  //   return x
  return x;
}

GPUFloatData directQR(CUSOLVERWrapper& cusolver, const GPUFloatData& A,
                      size_t rows, size_t columns, const GPUFloatData& b) {
  auto t1 = std::chrono::high_resolution_clock::now();
  // cusolver works inplace -> copy A and b
  GPUFloatData x = b;
  GPUFloatData QR = A;
  cusolver.solveDnQR(QR, static_cast<int>(rows), static_cast<int>(columns), x);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto passed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  VLOG(2) << "solveDnQR done after " << passed_time;
  return x;
}

void QRLS_inplace(CUSOLVERWrapper& cusolver, GPUFloatData& A, size_t rows,
                  size_t columns, GPUFloatData& b) {
  // cusolver works inplace -> copy A and b
  auto t1 = std::chrono::high_resolution_clock::now();
  cusolver.solveDnQR(A, static_cast<int>(rows), static_cast<int>(columns), b);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto passed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  VLOG(2) << "solveDnQR done after " << passed_time;
}

}  // namespace cu3d
