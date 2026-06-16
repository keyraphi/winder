cmake_minimum_required(VERSION 3.25)
project(winder VERSION 0.2.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Enable CUDA
enable_language(CUDA)
set(CUDA_ARCHITECTURES 
    75-real      # Turing (RTX 20-series, T4)
    80-real      # Ampere Data Center (A100)
    86-real      # Ampere Consumer (RTX 30-series)
    89-real      # Ada Lovelace (RTX 40-series, L4)
    90-real      # Hopper (H100)
    120-real     # Blackwell Base (B200)
    121-real     # Blackwell Extended / Spark Architecture (GB10)
    121-virtual  # Future-proof PTX blanket layer
)
# set(CUDA_ARCHITECTURES 120 89) # TEMPORAYR for faster compilation
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Include nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.10.2
    SYSTEM
)
FetchContent_MakeAvailable(nanobind)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src/cpp)
