#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

struct Mat3x3 {
  float data[9];
};

struct Mat3x3_bf16 {
  nv_bfloat16 data[9];
};
