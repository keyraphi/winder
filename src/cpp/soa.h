#pragma once

#include <cstddef>

template <typename T> struct SoAView {
  const float *base_ptr;
  size_t stride;
};
