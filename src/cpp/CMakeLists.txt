# Single shared library for the Python extension
nanobind_add_module(winder_backend
    winder.cpp
    winder_cuda.cu
    utils.cu
)

# CUDA compilation flags
target_compile_options(
    winder_backend
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall
        -Wextra
        -Werror>
        # CUDA-specific flags
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-res-usage>
        # Pass -Werror to the host compiler even when compiling .cu files
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)

# Link dependencies
target_link_libraries(winder_backend PRIVATE CUDA::cudart)
# C++ standard
target_compile_features(winder_backend PRIVATE cxx_std_17)

# Check for a DEV_MODE environment variable to add verbose ptxas info
if(DEFINED ENV{WINDER_DEV_MODE})
    message(STATUS "Winder: Development mode enabled (Verbose PTX enabled)")
    target_compile_options(
        winder_backend
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-res-usage>
    )
endif()

# for clangd
target_include_directories(winder_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# RPATH for Unix
if(UNIX)
    set_target_properties(winder_backend PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS winder_backend LIBRARY DESTINATION winder)
