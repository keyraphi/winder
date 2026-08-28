import ctypes
import os
import sys
import numpy as np


# Load CUDA Runtime Library
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
    raise RuntimeError(
        "Could not find CUDA Runtime library (libcudart / cudart64). Ensure CUDA toolkit is installed."
    )


_cudart = _load_cuda_runtime()

# Ctypes Signatures
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int
_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int
_cudart.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
_cudart.cudaMemcpyAsync.restype = ctypes.c_int
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamSynchronize.restype = ctypes.c_int

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


# DLPack Struct Definitions
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


DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorDeleter),
]


class CudaBuffer:
    """Manages GPU CUDA allocations with native DLPack export for winder execution."""

    def __init__(self, shape, dtype=np.float32, device_id=0):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.itemsize = self.dtype.itemsize
        self.nbytes = self.size * self.itemsize
        self.device_id = device_id

        ptr = ctypes.c_void_p()
        res = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(self.nbytes))
        if res != 0:
            raise MemoryError(f"cudaMalloc failed with code {res}")
        self.ptr = ptr

        # DLPack shape & stride arrays
        self._shape_arr = (ctypes.c_int64 * self.ndim)(*self.shape)
        strides = []
        cur = 1
        for s in reversed(self.shape):
            strides.append(cur)
            cur *= s
        self._strides_arr = (ctypes.c_int64 * self.ndim)(*reversed(strides))

    def copy_from_numpy_async(self, host_arr: np.ndarray, stream=None):
        assert host_arr.nbytes == self.nbytes, "Array byte size mismatch"
        stream_ptr = ctypes.c_void_p(stream) if stream else None
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
        stream_ptr = ctypes.c_void_p(stream) if stream else None
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

    def synchronize(self, stream=None):
        stream_ptr = ctypes.c_void_p(stream) if stream else None
        _cudart.cudaStreamSynchronize(stream_ptr)

    def __dlpack__(self, stream=None):
        managed = DLManagedTensor()
        managed.dl_tensor.data = self.ptr
        managed.dl_tensor.device = DLDevice(2, self.device_id)  # 2 = kDLCUDA
        managed.dl_tensor.ndim = self.ndim

        # Datatype conversion
        if self.dtype == np.float32:
            managed.dl_tensor.dtype_code, managed.dl_tensor.dtype_bits = 2, 32  # Float
        elif self.dtype == np.uint32:
            managed.dl_tensor.dtype_code, managed.dl_tensor.dtype_bits = 1, 32  # UInt
        elif self.dtype == np.int32:
            managed.dl_tensor.dtype_code, managed.dl_tensor.dtype_bits = 0, 32  # Int

        managed.dl_tensor.dtype_lanes = 1
        managed.dl_tensor.shape = self._shape_arr
        managed.dl_tensor.strides = self._strides_arr
        managed.dl_tensor.byte_offset = 0

        # Memory management callback
        py_managed = ctypes.pointer(managed)

        PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            PyCapsule_Destructor,
        ]
        PyCapsule_New.restype = ctypes.py_object

        return PyCapsule_New(py_managed, b"dltensor", PyCapsule_Destructor(0))

    def __dlpack_device__(self):
        return (2, self.device_id)

    def __del__(self):
        if hasattr(self, "ptr") and self.ptr:
            _cudart.cudaFree(self.ptr)
            self.ptr = None
