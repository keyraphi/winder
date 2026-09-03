import ctypes
import sys
import numpy as np

# ----------------------------------------------------------------------
# Load CUDA Runtime Library
# ----------------------------------------------------------------------
def _load_cuda_runtime():
    lib_names = (
        ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
        if sys.platform != "win32"
        else ["cudart64_12.dll", "cudart64_110.dll"]
    )
    for name in lib_names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("Could not find CUDA Runtime library (libcudart / cudart64).")

_cudart = _load_cuda_runtime()

# ----------------------------------------------------------------------
# CUDA function prototypes
# ----------------------------------------------------------------------
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int
_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int

_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamCreate.restype = ctypes.c_int
_cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamDestroy.restype = ctypes.c_int
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamSynchronize.restype = ctypes.c_int

_cudart.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
_cudart.cudaMemcpyAsync.restype = ctypes.c_int

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# ----------------------------------------------------------------------
# Helper to resolve stream argument
# ----------------------------------------------------------------------
def _resolve_stream_ptr(stream):
    """Convert CudaStream, int handle, or void_p into a ctypes.c_void_p."""
    if stream is None:
        return ctypes.c_void_p(0)
    if isinstance(stream, CudaStream):
        return stream.ptr
    if isinstance(stream, int):
        return ctypes.c_void_p(stream)
    if isinstance(stream, ctypes.c_void_p):
        return stream
    raise TypeError(f"Unsupported stream object type: {type(stream)}")


import ctypes


class CudaStream:
    """Encapsulates a CUDA stream handle and manages the lifetime of dependent GPU resources."""

    def __init__(self, device_id: int | None = None):
        self.ptr = ctypes.c_void_p()
        self._keep_alive = []

        if device_id is not None and _cudart is not None:
            _cudart.cudaSetDevice(device_id)

        dev = ctypes.c_int()
        res = _cudart.cudaGetDevice(ctypes.byref(dev)) if _cudart else 1
        self.device_id = dev.value if res == 0 else 0

        if _cudart:
            res = _cudart.cudaStreamCreate(ctypes.byref(self.ptr))
            if res != 0:
                raise RuntimeError(f"cudaStreamCreate failed with code {res}")

    @property
    def handle(self) -> int:
        return (
            self.ptr.value
            if (hasattr(self, "ptr") and self.ptr and self.ptr.value)
            else 0
        )

    def keep_alive(self, *objs):
        """Registers objects (engines, buffers, refs) to keep alive until stream exit."""
        for obj in objs:
            if isinstance(obj, (list, tuple)):
                self._keep_alive.extend(obj)
            elif obj is not None:
                self._keep_alive.append(obj)

    def _set_active_device(self) -> int:
        prev = ctypes.c_int()
        if _cudart is not None and _cudart.cudaGetDevice(ctypes.byref(prev)) != 0:
            prev.value = -1

        if (
            prev.value >= 0
            and prev.value != self.device_id
            and _cudart is not None
        ):
            _cudart.cudaSetDevice(self.device_id)
        return prev.value

    def _restore_device(self, prev_device: int):
        if (
            _cudart is not None
            and prev_device >= 0
            and prev_device != self.device_id
        ):
            _cudart.cudaSetDevice(prev_device)

    def synchronize(self):
        if (
            _cudart is not None
            and getattr(self, "ptr", None) is not None
            and self.ptr.value
        ):
            prev = self._set_active_device()
            _cudart.cudaStreamSynchronize(self.ptr)
            self._restore_device(prev)

    def destroy(self):
        """Explicitly destroys the CUDA stream handle."""
        if (
            _cudart is not None
            and getattr(self, "ptr", None) is not None
            and self.ptr.value
        ):
            prev = self._set_active_device()
            print("DEBUG: CudaStream.destroy: destroying stream")
            _cudart.cudaStreamDestroy(self.ptr)
            self._restore_device(prev)
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("DEBUG: CudaStream.__exit__: synchronizing stream")
        self.synchronize()

        print("DEBUG: CudaStream.__exit__: releasing keep_alive")
        self._keep_alive.clear()

        print("DEBUG: CudaStream.__exit__: destroying stream")
        self.destroy()

    def __del__(self):
        if (
            _cudart is not None
            and getattr(self, "ptr", None) is not None
            and self.ptr.value
        ):
            try:
                prev = self._set_active_device()
                print("DEBUG: CudaStream.__del__: destroying stream")
                _cudart.cudaStreamDestroy(self.ptr)
                self._restore_device(prev)
            except Exception:
                pass
            self.ptr = None

# ----------------------------------------------------------------------
# DLPack structure definitions
# ----------------------------------------------------------------------
class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype_code", ctypes.c_uint8),
        ("dtype_bits", ctypes.c_uint8),
        ("dtype_lanes", ctypes.c_uint16),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

class DLManagedTensor(ctypes.Structure):
    pass

# The deleter type: a C function pointer taking DLManagedTensor* and returning void.
DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))

DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorDeleter),
]

# ----------------------------------------------------------------------
# DLPack dtype mapping
# ----------------------------------------------------------------------
DTYPE_MAP = {
    np.dtype(np.float32): (2, 32),
    np.dtype(np.float64): (2, 64),
    np.dtype(np.float16): (2, 16),
    np.dtype(np.int32): (0, 32),
    np.dtype(np.int64): (0, 64),
    np.dtype(np.int16): (0, 16),
    np.dtype(np.int8): (0, 8),
    np.dtype(np.uint32): (1, 32),
    np.dtype(np.uint64): (1, 64),
    np.dtype(np.uint16): (1, 16),
    np.dtype(np.uint8): (1, 8),
    np.dtype(bool): (1, 8),
}

# ----------------------------------------------------------------------
# Python C API prototypes
# ----------------------------------------------------------------------
_pyapi = ctypes.pythonapi
_pyapi.PyCapsule_New.restype = ctypes.py_object
_pyapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# ----------------------------------------------------------------------
# CudaBuffer
# ----------------------------------------------------------------------
class CudaBuffer:
    """Manages GPU CUDA allocations with DLPack export."""
    def __init__(self, shape, dtype=np.float32, device_id=0):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape)) if self.shape else 1
        self.itemsize = self.dtype.itemsize
        self.nbytes = self.size * self.itemsize
        self.device_id = device_id

        ptr = ctypes.c_void_p()
        res = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(self.nbytes))
        if res != 0:
            raise MemoryError(f"cudaMalloc failed with code {res}")
        self.ptr = ptr

        # Keep references to DLPack buffers alive as long as this CudaBuffer exists.
        self._dlpack_buffers = []

    def copy_from_numpy_async(self, host_arr: np.ndarray, stream=None):
        assert host_arr.nbytes == self.nbytes, "Array byte size mismatch"
        stream_ptr = _resolve_stream_ptr(stream)
        host_ptr = host_arr.ctypes.data_as(ctypes.c_void_p)
        res = _cudart.cudaMemcpyAsync(
            self.ptr,
            host_ptr,
            ctypes.c_size_t(self.nbytes),
            cudaMemcpyHostToDevice,
            stream_ptr,
        )
        if res != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D failed with code {res}")

    def copy_to_numpy_async(self, host_arr: np.ndarray, stream=None):
        assert host_arr.nbytes == self.nbytes, "Array byte size mismatch"
        stream_ptr = _resolve_stream_ptr(stream)
        host_ptr = host_arr.ctypes.data_as(ctypes.c_void_p)
        res = _cudart.cudaMemcpyAsync(
            host_ptr,
            self.ptr,
            ctypes.c_size_t(self.nbytes),
            cudaMemcpyDeviceToHost,
            stream_ptr,
        )
        if res != 0:
            raise RuntimeError(f"cudaMemcpyAsync D2H failed with code {res}")

    def __dlpack__(self, stream=None):
        """Export a DLPack capsule. The consumer must NOT free the memory."""
        header_size = ctypes.sizeof(DLManagedTensor)
        array_size = self.ndim * ctypes.sizeof(ctypes.c_int64)
        total_bytes = header_size + 2 * array_size

        # Allocate a Python buffer and keep it alive via self._dlpack_buffers.
        buf = ctypes.create_string_buffer(total_bytes)
        raw_mem = ctypes.addressof(buf)

        managed_struct = DLManagedTensor.from_address(raw_mem)
        shape_ptr = ctypes.cast(raw_mem + header_size, ctypes.POINTER(ctypes.c_int64))
        strides_ptr = ctypes.cast(
            raw_mem + header_size + array_size, ctypes.POINTER(ctypes.c_int64)
        )

        # Fill shape
        for i, s in enumerate(self.shape):
            shape_ptr[i] = s

        # Compute contiguous C-order strides (in elements)
        strides = [1] * self.ndim
        for i in range(self.ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        for i, st in enumerate(strides):
            strides_ptr[i] = st

        managed_struct.dl_tensor.data = self.ptr
        managed_struct.dl_tensor.device = DLDevice(2, self.device_id)  # kDLCUDA = 2
        managed_struct.dl_tensor.ndim = self.ndim

        if self.dtype in DTYPE_MAP:
            code, bits = DTYPE_MAP[self.dtype]
            managed_struct.dl_tensor.dtype_code = code
            managed_struct.dl_tensor.dtype_bits = bits
        else:
            raise ValueError(f"Unsupported DLPack dtype: {self.dtype}")

        managed_struct.dl_tensor.dtype_lanes = 1
        managed_struct.dl_tensor.shape = shape_ptr
        managed_struct.dl_tensor.strides = strides_ptr
        managed_struct.dl_tensor.byte_offset = 0
        managed_struct.manager_ctx = None

        # Set deleter to NULL: consumer must not free this memory.
        managed_struct.deleter = DLManagedTensorDeleter()

        # Create the capsule with a NULL destructor.
        capsule = _pyapi.PyCapsule_New(raw_mem, b"dltensor", ctypes.c_void_p(0))
        if not capsule:
            raise RuntimeError("Failed to create DLPack capsule")

        # Keep the buffer alive as long as this CudaBuffer exists.
        self._dlpack_buffers.append(buf)

        return capsule

    def __dlpack_device__(self):
        return (2, self.device_id)

    def __del__(self):
        if hasattr(self, "ptr") and self.ptr:
            _cudart.cudaFree(self.ptr)
            self.ptr = None
