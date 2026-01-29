cmake_minimum_required(VERSION 3.18)
project(winder VERSION 0.2.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Enable CUDA
enable_language(CUDA)
# set(CUDA_ARCHITECTURES 75 80 86 89 90)
set(CUDA_ARCHITECTURES 75 80 90) # TEMPORAYR for faster compilation
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Include nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.10.2
)
FetchContent_MakeAvailable(nanobind)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src/cpp)
