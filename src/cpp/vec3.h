#pragma once
#include <cuda_runtime_api.h>
#include <vector_types.h>

struct Vec3 {
    float x, y, z;

    __host__ __device__ __forceinline__ Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    __host__ __device__ __forceinline__ Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    __host__ __device__ __forceinline__ Vec3 operator*(Vec3 b) const { return {x * b.x, y * b.y, z * b.z}; }
    __host__ __device__ __forceinline__ Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

