#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <optional>
#include <string>

namespace nb = nanobind;

using Scalar_t = nb::ndarray<nb::array_api, float, nb::shape<-1>, nb::c_contig,
                             nb::device::cuda>;

namespace winder_cuda {

// Forward declarations
class ScopedCudaDevice;

// CUDA memory management functions
void *cuda_allocate(size_t size);
void cuda_free(void *ptr);
bool cuda_memcpy(void *dest, void *src, size_t bytes);

Scalar_t scalar_with_default(const ::std::optional<Scalar_t> &maybe_pc_wn,
                             size_t count, float default_value);

} // namespace cuda
