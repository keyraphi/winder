#pragma once

#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// For leaf nodes
// 60 byte
struct alignas(64) TailorCoefficientsF16 {
  Vec3_f16 zero_order;
  Mat3x3_f16 first_order;
  Tensor3_bf16_compressed second_order;
};

// For m2m
// 120 byte
struct TailorCoefficients {
  Vec3 zero_order;
  Mat3x3 first_order;
  Tensor3_compressed second_order;

  __host__ __device__ static auto
  from_f16(const TailorCoefficientsF16 &t) -> TailorCoefficients {
    TailorCoefficients result;
    result.zero_order = Vec3::from_f16(t.zero_order);
    result.first_order = Mat3x3::from_f16(t.first_order);
    result.second_order = Tensor3_compressed::from_f16(t.second_order);
    return result;
  }
};
