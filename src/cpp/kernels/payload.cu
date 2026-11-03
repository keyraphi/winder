#include "kernels/common.cuh"
#include "kernels/payload.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdexcept>

template <uint32_t D, uint32_t BlockSize = 256>
__global__ void gather_payload_kernel(const float *__restrict__ payload,
                                      const uint32_t *__restrict__ sort_indices,
                                      float *__restrict__ sorted_payload,
                                      size_t total_elements) {

  const size_t idx = static_cast<size_t>(blockIdx.x) * BlockSize + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  const size_t row = idx / D;
  const size_t col = idx - row * D;

  const uint32_t src_row = __ldg(sort_indices + row);
  sorted_payload[idx] = __ldg(payload + static_cast<size_t>(src_row) * D + col);
}

template <uint32_t D>
void launch_gather_payload(const float *payload, const uint32_t *sort_indices,
                           float *sorted_payload, size_t n_rows,
                           cudaStream_t stream) {

  const size_t total = n_rows * static_cast<size_t>(D);
  if (total == 0) {
    return;
  }

  constexpr uint32_t threads = 256;

  const auto blocks =
      static_cast<uint32_t>((total + threads - 1) / threads);

  gather_payload_kernel<D, threads><<<blocks, threads, 0, stream>>>(
      payload, sort_indices, sorted_payload, total);
  CUDA_CHECK(cudaGetLastError());
}

template <uint32_t D>
void gather_payload(const float *payload, const uint32_t *sort_indices,
                    float *sorted_payload, size_t n_rows, cudaStream_t stream) {
  launch_gather_payload<D>(payload, sort_indices, sorted_payload, n_rows,
                           stream);
}

auto payload_d_bucket(uint32_t d) -> uint32_t {
  if (d == 0) {
    throw std::runtime_error("Payload dimensionality D must be > 0");
  }
  static constexpr uint32_t kBuckets[] = {1, 2, 3, 4, 8, 16, 32, 64, 128};
  for (uint32_t b : kBuckets) {
    if (b >= d) {
      return b;
    }
  }
  throw std::runtime_error(
      "Payload dimensionality D > 128 is not supported by the "
      "PayloadWindingNumbersBackend");
}

void gather_payload_dynamic(const float *payload, const uint32_t *sort_indices,
                            float *sorted_payload, size_t n_rows, uint32_t d,
                            cudaStream_t stream) {
  switch (payload_d_bucket(d)) {
  case 1:
    return gather_payload<1>(payload, sort_indices, sorted_payload, n_rows,
                             stream);
  case 2:
    return gather_payload<2>(payload, sort_indices, sorted_payload, n_rows,
                             stream);
  case 3:
    return gather_payload<3>(payload, sort_indices, sorted_payload, n_rows,
                             stream);
  case 4:
    return gather_payload<4>(payload, sort_indices, sorted_payload, n_rows,
                             stream);
  case 8:
    return gather_payload<8>(payload, sort_indices, sorted_payload, n_rows,
                             stream);
  case 16:
    return gather_payload<16>(payload, sort_indices, sorted_payload, n_rows,
                              stream);
  case 32:
    return gather_payload<32>(payload, sort_indices, sorted_payload, n_rows,
                              stream);
  case 64:
    return gather_payload<64>(payload, sort_indices, sorted_payload, n_rows,
                              stream);
  case 128:
    return gather_payload<128>(payload, sort_indices, sorted_payload, n_rows,
                               stream);
  default:
    // Unreachable: payload_d_bucket either returns one of the above or throws
    throw std::runtime_error(
        "Internal error: payload_d_bucket returned an unexpected value");
  }
}

// ---------------------------------------------------------------------------
// Explicit instantiations — one per D bucket.
template void gather_payload<1>(const float *, const uint32_t *, float *,
                                size_t, cudaStream_t);
template void gather_payload<2>(const float *, const uint32_t *, float *,
                                size_t, cudaStream_t);
template void gather_payload<3>(const float *, const uint32_t *, float *,
                                size_t, cudaStream_t);
template void gather_payload<4>(const float *, const uint32_t *, float *,
                                size_t, cudaStream_t);
template void gather_payload<8>(const float *, const uint32_t *, float *,
                                size_t, cudaStream_t);
template void gather_payload<16>(const float *, const uint32_t *, float *,
                                 size_t, cudaStream_t);
template void gather_payload<32>(const float *, const uint32_t *, float *,
                                 size_t, cudaStream_t);
template void gather_payload<64>(const float *, const uint32_t *, float *,
                                 size_t, cudaStream_t);
template void gather_payload<128>(const float *, const uint32_t *, float *,
                                  size_t, cudaStream_t);
