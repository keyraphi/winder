# Keep kernels separate, speeding up incremental builds.
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu")
add_library(winder_kernels STATIC ${KERNEL_SOURCES})
add_library(winder_backend STATIC winder_cuda.cu utils.cu)
# make sure to use position independent code (pic)
set_target_properties(winder_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(winder_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Cudda Compilation Flags
set(WINDER_CUDA_FLAGS
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)

target_compile_options(winder_kernels PRIVATE ${WINDER_CUDA_FLAGS})
target_compile_options(winder_backend PRIVATE ${WINDER_CUDA_FLAGS})

target_link_libraries(winder_backend PRIVATE winder_kernels)

# Handle Include Directories
target_include_directories(winder_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(winder_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Main nanobind Module
nanobind_add_module(winder_module
    winder.cpp
)

# Link and Configure Module
target_link_libraries(
    winder_module
    PRIVATE winder_kernels winder_backend CUDA::cudart
)
target_compile_options(winder_module PRIVATE ${WINDER_CUDA_FLAGS})

# for clangd and backend host code
target_include_directories(winder_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Define the option (OFF by default)
option(WINDER_DEV_MODE "Enable development mode (verbose PTX info)" OFF)

if(WINDER_DEV_MODE)
    message(STATUS "Winder: Development mode enabled")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
    set(CUDA_SEPARATE_COMPILATION ON)

    set(CUDA_BASE_DEV_FLAGS "-res-usage" "-lineinfo" "-Xcompiler" "-rdynamic")

    target_compile_options(
        winder_kernels
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_BASE_DEV_FLAGS}>
    )
    target_compile_options(
        winder_backend
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_BASE_DEV_FLAGS}>
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(
            STATUS
            "Winder: CUDA Debugging enabled (Optimizations disabled)"
        )
        add_definitions(-DDEBUG)
        set(CUDA_DEBUG_ONLY_FLAGS "-G" "-g")

        target_compile_options(
            winder_kernels
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_ONLY_FLAGS}>
        )
        target_compile_options(
            winder_backend
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_ONLY_FLAGS}>
        )
    endif()
endif()

# Install logic
if(UNIX)
    set_target_properties(winder_module PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS winder_module LIBRARY DESTINATION winder)
