#!/bin/bash
# Clean artifacts
rm -rf build/

# Run with LOCAL image override
CIBW_MANYLINUX_X86_64_IMAGE="manylinux-cuda-dev" \
CIBW_BUILD="cp312-manylinux_x86_64" \
python -m cibuildwheel --output-dir wheelhouse
