#!/bin/bash
set -e -x

# Clear out old builds
rm -rf /io/dist /io/wheelhouse /io/build

#  Dynamically find and loop through all modern Python versions (3.10 and newer)
# The glob 'cp31*' matches cp310, cp311, cp312, cp313, cp314, etc.
for PYDIR in /opt/python/cp31*; do
    # Ensure it's actually a directory before proceeding
    if [ -d "$PYDIR" ]; then
        PYBIN="${PYDIR}/bin"
        echo "========================================================"
        echo "Building wheel using: ${PYBIN}/python"
        echo "========================================================"
        
        # Upgrade packaging tools inside this specific Python environment
        "${PYBIN}/pip" install --upgrade pip build setuptools wheel
        
        # Build the raw wheel
        "${PYBIN}/pip" wheel /io/ --no-deps -w /io/dist/
    fi
done

# Repair the wheels using auditwheel
echo "Repairing wheels to make them manylinux compliant..."
for whl in /io/dist/*.whl; do
    # Exclude libcuda.so.1 so it links to the host system's NVIDIA driver
    auditwheel repair "$whl" \
        --plat manylinux_2_28_x86_64 \
        -w /io/wheelhouse/ \
        -e libcuda.so.1
done

echo "Successfully built and repaired wheels in 'wheelhouse/'!"
