#pragma once

#include "mat3x3.h"
#include "tensor3.h"
#include "vec3.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

// For leaf nodes
// 60 byte aligned to 64 byte
struct alignas(64) TailorCoefficientsF16 {
  Vec3_f16 zero_order;
  Mat3x3_f16 first_order;
  Tensor3_bf16_compressed second_order;
};

// For m2m
// 120 byte aligned to 128 bytes
struct alignas(128) TailorCoefficients {
  Vec3 zero_order;
  Mat3x3 first_order;
  Tensor3_compressed second_order;

  __host__ __device__ static auto from_f16(const TailorCoefficientsF16 &t)
      -> TailorCoefficients {
    TailorCoefficients result;
    result.zero_order = Vec3::from_f16(t.zero_order);
    result.first_order = Mat3x3::from_f16(t.first_order);
    result.second_order = Tensor3_compressed::from_f16(t.second_order);
    return result;
  }
};

// ###### BACKWARD ########################
// 26 byte content aligned to 32 byte
struct alignas(32) BackwardTailorCoefficientsF16 {
  half zero_order;
  Vec3_f16 first_order;
  Mat3x3_f16 second_order;
};

// For m2m
// 53 byte aligned to 64 byte
struct alignas(64) BackwardTailorCoefficients {
  float zero_order;
  Vec3 first_order;
  Mat3x3 second_order;

  __host__ __device__ static auto
  from_f16(const BackwardTailorCoefficientsF16 &t)
      -> BackwardTailorCoefficients {
    BackwardTailorCoefficients result;
    result.zero_order = __half2float(t.zero_order);
    result.first_order = Vec3::from_f16(t.first_order);
    result.second_order = Mat3x3::from_f16(t.second_order);
    return result;
  }
};
