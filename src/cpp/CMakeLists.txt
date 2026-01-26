# Keep kernels separate, speeding up incremental builds.
file(GLOB KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu")
add_library(winder_kernels STATIC ${KERNEL_SOURCES})
# make sure to use position independent code (pic)
set_target_properties(winder_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Cudda Compilation Flags
set(WINDER_CUDA_FLAGS
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)

target_compile_options(winder_kernels PRIVATE ${WINDER_CUDA_FLAGS})
target_compile_features(winder_kernels PUBLIC cxx_std_17)

# Handle Include Directories
target_include_directories(winder_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Main nanobind Module
nanobind_add_module(winder_backend
    winder.cpp
    winder_cuda.cu
    utils.cu
)

# Link and Configure Module
target_link_libraries(winder_backend PRIVATE winder_kernels CUDA::cudart)
target_compile_options(winder_backend PRIVATE ${WINDER_CUDA_FLAGS})

# for clangd and backend host code
target_include_directories(winder_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED ENV{WINDER_DEV_MODE})
    message(STATUS "Winder: Development mode enabled (Verbose PTX enabled)")
    target_compile_options(
        winder_kernels
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-res-usage>
    )
    target_compile_options(
        winder_backend
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-res-usage>
    )
endif()

# Install logic
if(UNIX)
    set_target_properties(winder_backend PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS winder_backend LIBRARY DESTINATION winder)

# Unit Tests
add_executable(test_quantization tests/test_quantization.cu)
target_link_libraries(test_quantization PRIVATE winder_kernels)
target_compile_options(test_quantization PRIVATE ${WINDER_CUDA_FLAGS})

FetchContent_Declare(
    googletest
    URL
        https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows compatibility with GTest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
    test_binary_tree_construction
    tests/test_binary_tree_construction.cu
)
add_executable(
    test_binary2bvh8
    tests/test_binary2bvh8.cu
)

target_link_libraries(
    test_binary_tree_construction
    PRIVATE winder_kernels GTest::gtest_main CUDA::cudart
)
target_link_libraries(
    test_binary2bvh8
    PRIVATE winder_kernels GTest::gtest_main CUDA::cudart
)

target_compile_options(
    test_binary_tree_construction
    PRIVATE ${WINDER_CUDA_FLAGS}
)
target_compile_options(
    test_binary2bvh8
    PRIVATE ${WINDER_CUDA_FLAGS}
)
target_include_directories(
    test_binary_tree_construction
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(
    test_binary2bvh8
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

enable_testing()
add_test(NAME BinaryTreeConstructionTest COMMAND test_binary_tree_construction)
add_test(NAME Binary2BVH8Test COMMAND test_binary2bvh8)
