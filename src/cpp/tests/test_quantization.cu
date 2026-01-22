#include "../mat3x3.h"
#include "../tailor_coefficients.cuh"
#include "../tailor_coefficients.h"
#include "../tensor3.h"
#include "../vec3.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <random>
#include <vector>

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

__global__ void test_TailorCoefficientQuantization_kernel(
    const TailorCoefficientsBf16 *tailor_in,
    TailorCoefficientsQuantized *tailor_quantized_out,
    TailorCoefficientsBf16 *tailor_out, float *shared_scale_out) {
  if (threadIdx.x > 0) {
    return;
  }
  Vec3_bf16 zero_bf16 = tailor_in[0].zero_order;
  Mat3x3_bf16 first_bf16 = tailor_in[0].first_order;
  Tensor3_bf16_compressed second_bf16 = tailor_in[0].second_order;

  Vec3 zero_f32 = Vec3::from_bf16(zero_bf16);
  Mat3x3 first_f32 = Mat3x3::from_bf16(first_bf16);
  Tensor3_compressed second_f32 = Tensor3_compressed::from_bf16(second_bf16);

  // Verify input data
  // printf("INPUT----------------------------------\n");
  // printf("zero_order input:\n [%f, %f, %f]\n\n", zero_f32.x, zero_f32.y,
  //        zero_f32.z);
  // printf("first_order input:\n");
  // printf("[[%f, %f, %f],\n", first_f32.data[0], first_f32.data[1],
  //        first_f32.data[2]);
  // printf(" [%f, %f, %f],\n", first_f32.data[3], first_f32.data[4],
  //        first_f32.data[5]);
  // printf(" [%f, %f, %f]]\n\n", first_f32.data[6], first_f32.data[7],
  //        first_f32.data[8]);
  // printf("second_order input:\n");
  // printf("[[[%f, %f, %f],\n", second_f32.data[0], second_f32.data[1],
  //        second_f32.data[2]);
  // printf("  [%f, %f, %f],\n", second_f32.data[3], second_f32.data[4],
  //        second_f32.data[5]);
  // printf("  [%f, %f, %f]],\n", second_f32.data[6], second_f32.data[7],
  //        second_f32.data[8]);
  // printf(" [[%f, %f, %f],\n", second_f32.data[9], second_f32.data[10],
  //        second_f32.data[11]);
  // printf("  [%f, %f, %f],\n", second_f32.data[12], second_f32.data[13],
  //        second_f32.data[14]);
  // printf("  [%f, %f, %f]],\n", second_f32.data[15], second_f32.data[16],
  //        second_f32.data[17]);
  // printf("---------------------------------------\n\n\n");

  // quantize
  TailorCoefficientsQuantized tailor_q;
  tailor_q.set_tailor_coefficients(zero_f32, first_f32, second_f32);
  tailor_quantized_out[0] = tailor_q;

  // unpack
  TailorCoefficientsBf16 tailor;
  float shared_scale_factor = tailor_q.get_shared_scale_factor();
  // printf("SHARED SCALE FACTOR\n");
  // printf("shared_exponent: %u\n", tailor_q.tailor_data[43]);
  // printf("shared_scale_factor: %f\n", shared_scale_factor);
  // printf(
  //     "quantized coefficients: %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u\n",
  //     tailor_q.tailor_data[0], tailor_q.tailor_data[1],
  //     tailor_q.tailor_data[2], tailor_q.tailor_data[3],
  //     tailor_q.tailor_data[4], tailor_q.tailor_data[5],
  //     tailor_q.tailor_data[6], tailor_q.tailor_data[7],
  //     tailor_q.tailor_data[8], tailor_q.tailor_data[9],
  //     tailor_q.tailor_data[10]);
  // printf("tailor_data[7] hex: 0x%08X\n", tailor_q.tailor_data[7]);
  shared_scale_out[0] = shared_scale_factor;
  tailor.zero_order = tailor_q.get_tailor_zero_order(shared_scale_factor);
  tailor.first_order = tailor_q.get_tailor_first_order(shared_scale_factor);
  tailor.second_order = tailor_q.get_tailor_second_order(shared_scale_factor);
  tailor_out[0] = tailor;

  // Verify values after quantization
  zero_f32 = tailor.zero_order;
  first_f32 = tailor.first_order;
  second_f32 = tailor.second_order;

  // Verify output data
  // printf("OUTPUT----------------------------------\n");
  // printf("zero_order output:\n [%f, %f, %f]\n\n", zero_f32.x, zero_f32.y,
  //        zero_f32.z);
  // printf("first_order output:\n");
  // printf("[[%f, %f, %f],\n", first_f32.data[0], first_f32.data[1],
  //        first_f32.data[2]);
  // printf(" [%f, %f, %f],\n", first_f32.data[3], first_f32.data[4],
  //        first_f32.data[5]);
  // printf(" [%f, %f, %f]]\n\n", first_f32.data[6], first_f32.data[7],
  //        first_f32.data[8]);
  // printf("second_order output:\n");
  // printf("[[[%f, %f, %f],\n", second_f32.data[0], second_f32.data[1],
  //        second_f32.data[2]);
  // printf("  [%f, %f, %f],\n", second_f32.data[3], second_f32.data[4],
  //        second_f32.data[5]);
  // printf("  [%f, %f, %f]],\n", second_f32.data[6], second_f32.data[7],
  //        second_f32.data[8]);
  // printf(" [[%f, %f, %f],\n", second_f32.data[9], second_f32.data[10],
  //        second_f32.data[11]);
  // printf("  [%f, %f, %f],\n", second_f32.data[12], second_f32.data[13],
  //        second_f32.data[14]);
  // printf("  [%f, %f, %f]],\n", second_f32.data[15], second_f32.data[16],
  //        second_f32.data[17]);
  // printf("----------------------------------------\n");
}

// Helper to get random floats
float get_rand(float min, float max, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(min, max);
  return dist(gen);
}

// Fills the struct with random values and returns the max absolute value found
float fill_random(TailorCoefficientsBf16 &host_struct, float range,
                  std::mt19937 &gen) {
  float max_abs = 0.0F;
  auto fill_val = [&](nv_bfloat16 &target) {
    float val = get_rand(-range, range, gen);
    max_abs = std::max(max_abs, std::abs(val));
    target = __float2bfloat16(val);
  };

  fill_val(host_struct.zero_order.x);
  fill_val(host_struct.zero_order.y);
  fill_val(host_struct.zero_order.z);
  for (auto &v : host_struct.first_order.data)
    fill_val(v);
  for (auto &v : host_struct.second_order.data)
    fill_val(v);

  return max_abs;
}

bool check_close(const char *label, nv_bfloat16 actual_bf, float original_f32,
                 float threshold) {
  float actual_f32 = __bfloat162float(actual_bf);
  float diff = std::abs(actual_f32 - original_f32);
  if (diff > threshold + 1e-5f) { // adding small epsilon for bf16 precision
    printf("FAIL [%s]: Orig: %f, Got: %f, Diff: %f (Max Allowed: %f)\n", label,
           original_f32, actual_f32, diff, threshold);
    return false;
  }
  return true;
}

// Fills with specific edge-case patterns
void fill_edge_case(TailorCoefficientsBf16 &host_struct, int case_type) {
  auto set_all = [&](float val) {
    nv_bfloat16 b = __float2bfloat16(val);
    host_struct.zero_order.x = b;
    host_struct.zero_order.y = b;
    host_struct.zero_order.z = b;
    for (auto &v : host_struct.first_order.data)
      v = b;
    for (auto &v : host_struct.second_order.data)
      v = b;
  };

  switch (case_type) {
  case 0: // All Zeros
    set_all(0.0f);
    break;
  case 1: // Single Outlier (The "Needle in a Haystack")
    set_all(0.001f);
    host_struct.second_order.data[17] = __float2bfloat16(1000.0f);
    break;
  case 2: // All Negative
    set_all(-500.0f);
    break;
  case 3: // Power of 2 Boundary (Exactly 1024)
    set_all(1024.0f);
    break;
  case 4: // Negative Boundary (Exactly -2048 with max_abs 2048)
    set_all(-1024.0f);
    break;
  }
}

void run_edge_cases(TailorCoefficientsBf16 *d_in,
                    TailorCoefficientsQuantized *d_q,
                    TailorCoefficientsBf16 *d_out, float *d_scale) {
  const char *names[] = {"ALL ZEROS", "SINGLE OUTLIER", "ALL NEGATIVE",
                         "POW2 BOUNDARY", "NEGATIVE BOUNDARY"};

  for (int i = 0; i < 5; ++i) {
    printf("Testing Edge Case: %s... ", names[i]);

    TailorCoefficientsBf16 h_in, h_out;
    fill_edge_case(h_in, i);

    CUDA_CHECK(cudaMemcpy(d_in, &h_in, sizeof(TailorCoefficientsBf16),
                          cudaMemcpyHostToDevice));

    test_TailorCoefficientQuantization_kernel<<<1, 1>>>(d_in, d_q, d_out,
                                                        d_scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_scale;
    CUDA_CHECK(
        cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(TailorCoefficientsBf16),
                          cudaMemcpyDeviceToHost));

    // Validation logic
    float max_err = h_scale * 0.501f; // Slight margin for BF16
    bool success = true;

    // We check index 17 specifically for the outlier case
    float orig_17 = __bfloat162float(h_in.second_order.data[17]);
    float got_17 = __bfloat162float(h_out.second_order.data[17]);

    if (std::abs(orig_17 - got_17) > max_err)
      success = false;

    if (success)
      printf("PASSED (Scale: %e)\n", h_scale);
    else
      printf("FAILED! (Target: %f, Got: %f, Scale: %f)\n", orig_17, got_17,
             h_scale);
  }
}

int main() {
  // device stuff
  float *shared_scale_device;
  TailorCoefficientsBf16 *tailor_input_device;
  TailorCoefficientsQuantized *tailor_quantized_device;
  TailorCoefficientsBf16 *tailor_output_device;
  CUDA_CHECK(cudaMalloc(&shared_scale_device, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&tailor_input_device, sizeof(TailorCoefficientsBf16)));
  CUDA_CHECK(cudaMalloc(&tailor_quantized_device,
                        sizeof(TailorCoefficientsQuantized)));
  CUDA_CHECK(cudaMalloc(&tailor_output_device, sizeof(TailorCoefficientsBf16)));

  // random stuff
  std::mt19937 gen(42);
  std::vector<float> test_ranges = {1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f, 100000.0f, 1000000.0f};

  for (float range : test_ranges) {
    printf("Testing Range: [-%.1f, %.1f]... ", range, range);

    TailorCoefficientsBf16 input_host, output_host;
    float scale_host;

    // 1. Generate Data
    float max_val = fill_random(input_host, range, gen);

    // 2. GPU Processing
    CUDA_CHECK(cudaMemcpy(tailor_input_device, &input_host,
                          sizeof(TailorCoefficientsBf16),
                          cudaMemcpyHostToDevice));
    test_TailorCoefficientQuantization_kernel<<<1, 1>>>(
        tailor_input_device, tailor_quantized_device, tailor_output_device,
        shared_scale_device);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&scale_host, shared_scale_device, sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&output_host, tailor_output_device,
                          sizeof(TailorCoefficientsBf16),
                          cudaMemcpyDeviceToHost));

    // 3. Validation
    // The max error for rounding to nearest is scale/2.
    float max_allowed_error = scale_host * 0.5f;
    bool success = true;

    // Check a few samples or the whole set
    if (!check_close("zero", output_host.zero_order.x,
                     __bfloat162float(input_host.zero_order.x),
                     max_allowed_error))
      success = false;
    if (!check_close("zero", output_host.zero_order.y,
                     __bfloat162float(input_host.zero_order.y),
                     max_allowed_error))
      success = false;
    if (!check_close("zero", output_host.zero_order.z,
                     __bfloat162float(input_host.zero_order.z),
                     max_allowed_error))
      success = false;
    for (int i = 0; i < 9; ++i) {
      if (!check_close("first", output_host.first_order.data[i],
                       __bfloat162float(input_host.first_order.data[i]),
                       max_allowed_error))
        success = false;
    }
    for (int i = 0; i < 18; ++i) {
      if (!check_close("second", output_host.second_order.data[i],
                       __bfloat162float(input_host.second_order.data[i]),
                       max_allowed_error))
        success = false;
    }

    if (success) {
      printf("PASSED (Scale: %f, Max Err: %f)\n", scale_host,
             max_allowed_error);
    } else {
      printf("RANGE FAIL at %.1f\n", range);
      // break; // Stop if we find a logic error
    }
  }

  run_edge_cases(tailor_input_device, tailor_quantized_device,
                 tailor_output_device, shared_scale_device);

  // ... (cudaFree calls) ...
  cudaFree(shared_scale_device);
  cudaFree(tailor_input_device);
  cudaFree(tailor_quantized_device);
  cudaFree(tailor_output_device);
  return 0;
}

// int main() {
//
//   TailorCoefficientsBf16 tailor_input_host;
//   tailor_input_host.zero_order = {0.0, 0.1, 0.2};
//   tailor_input_host.first_order = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
//   0.8}; tailor_input_host.second_order = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
//                                     0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
//                                     1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
//   TailorCoefficientsQuantized tailor_quantized_host;
//   TailorCoefficientsBf16 tailor_out_host;
//   float shared_scale_host;
//
//   float *shared_scale_device;
//   TailorCoefficientsBf16 *tailor_input_device;
//   TailorCoefficientsQuantized *tailor_quantized_device;
//   TailorCoefficientsBf16 *tailor_output_device;
//
//   CUDA_CHECK(cudaMalloc(&shared_scale_device, sizeof(float)));
//   CUDA_CHECK(cudaMalloc(&tailor_input_device,
//   sizeof(TailorCoefficientsBf16)));
//   CUDA_CHECK(cudaMalloc(&tailor_quantized_device,
//                         sizeof(TailorCoefficientsQuantized)));
//   CUDA_CHECK(cudaMalloc(&tailor_output_device,
//   sizeof(TailorCoefficientsBf16)));
//   CUDA_CHECK(cudaMemcpy(tailor_input_device, &tailor_input_host,
//                         sizeof(TailorCoefficientsBf16),
//                         cudaMemcpyHostToDevice));
//   test_TailorCoefficientQuantization_kernel<<<1, 1>>>(
//       tailor_input_device, tailor_quantized_device, tailor_output_device,
//       shared_scale_device);
//   CUDA_CHECK(cudaGetLastError());
//   CUDA_CHECK(cudaDeviceSynchronize());
//   CUDA_CHECK(cudaMemcpy(&shared_scale_host, shared_scale_device,
//   sizeof(float),
//                         cudaMemcpyDeviceToHost));
//   CUDA_CHECK(cudaMemcpy(&tailor_quantized_host, tailor_quantized_device,
//                         sizeof(TailorCoefficientsQuantized),
//                         cudaMemcpyDeviceToHost));
//   CUDA_CHECK(cudaMemcpy(&tailor_out_host, tailor_output_device,
//                         sizeof(TailorCoefficientsBf16),
//                         cudaMemcpyDeviceToHost));
//
//   printf("HOST after download\n");
//   printf("shared_scale: %f\n", shared_scale_host);
//   printf("quantized coefficients: %u, %u, %u, %u, %u, %u, %u, %u, %u, %u,
//   %u\n",
//          tailor_quantized_host.tailor_data[0],
//          tailor_quantized_host.tailor_data[1],
//          tailor_quantized_host.tailor_data[2],
//          tailor_quantized_host.tailor_data[3],
//          tailor_quantized_host.tailor_data[4],
//          tailor_quantized_host.tailor_data[5],
//          tailor_quantized_host.tailor_data[6],
//          tailor_quantized_host.tailor_data[7],
//          tailor_quantized_host.tailor_data[8],
//          tailor_quantized_host.tailor_data[9],
//          tailor_quantized_host.tailor_data[10]);
//
//   Vec3 zero_f32 = Vec3::from_bf16(tailor_out_host.zero_order);
//   Mat3x3 first_f32 = Mat3x3::from_bf16(tailor_out_host.first_order);
//   Tensor3_compressed second_f32_compressed =
//       Tensor3_compressed::from_bf16(tailor_out_host.second_order);
//   Tensor3 second_f32 = second_f32_compressed.uncompress();
//   printf("zero_order output:\n [%f, %f, %f]\n\n", zero_f32.x, zero_f32.y,
//          zero_f32.z);
//   printf("first_order output:\n");
//   printf("[[%f, %f, %f],\n", first_f32.data[0], first_f32.data[1],
//          first_f32.data[2]);
//   printf(" [%f, %f, %f],\n", first_f32.data[3], first_f32.data[4],
//          first_f32.data[5]);
//   printf(" [%f, %f, %f]]\n\n", first_f32.data[6], first_f32.data[7],
//          first_f32.data[8]);
//   printf("second_order output:\n");
//   printf("[[[%f, %f, %f],\n", second_f32.data[0], second_f32.data[1],
//          second_f32.data[2]);
//   printf("  [%f, %f, %f],\n", second_f32.data[3], second_f32.data[4],
//          second_f32.data[5]);
//   printf("  [%f, %f, %f]],\n", second_f32.data[6], second_f32.data[7],
//          second_f32.data[8]);
//   printf(" [[%f, %f, %f],\n", second_f32.data[9], second_f32.data[10],
//          second_f32.data[11]);
//   printf("  [%f, %f, %f],\n", second_f32.data[12], second_f32.data[13],
//          second_f32.data[14]);
//   printf("  [%f, %f, %f]],\n", second_f32.data[15], second_f32.data[16],
//          second_f32.data[17]);
//   printf(" [[%f, %f, %f],\n", second_f32.data[18], second_f32.data[19],
//          second_f32.data[20]);
//   printf("  [%f, %f, %f],\n", second_f32.data[21], second_f32.data[22],
//          second_f32.data[23]);
//   printf("  [%f, %f, %f]]]\n", second_f32.data[24], second_f32.data[25],
//          second_f32.data[26]);
//
//   printf("Done\n");
//   return 0;
// }
