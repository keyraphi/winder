import time
import numpy as np
import math

from dlpack_bridge import CudaBuffer, CudaStream
import winder


def generate_sphere_point_cloud(num_points: int = 1000, radius: float = 1.0):
    """Generates synthetic points and oriented normals on a 3D sphere."""
    indices = np.arange(0, num_points, dtype=np.float32) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    points = np.stack([x, y, z], axis=1).astype(np.float32) * radius
    area_per_point = (4.0 * np.pi * (radius**2)) / num_points
    scaled_normals = points / radius * area_per_point

    return points, scaled_normals.astype(np.float32)


def compute_winding_numbers_numpy(points, normals, queries, epsilon):
    """
    Reference winding number calculation for point clouds using NumPy.
    Mirrors the PyTorch code provided.
    """
    inv_epsilon = 1.0 / epsilon
    four_over_3sqrt_pi = 4.0 / (3.0 * math.sqrt(math.pi))
    near_limit_constant = four_over_3sqrt_pi * (inv_epsilon**3)
    inv_four_pi = 1.0 / (4.0 * math.pi)

    p = points  # (N, 3)
    n = normals  # (N, 3)
    q = queries  # (M, 3)

    d = p[None, :, :] - q[:, None, :]  # (M, N, 3)
    dist2 = np.sum(d * d, axis=-1)  # (M, N)
    inv_distance = 1.0 / np.sqrt(dist2 + 1e-30)
    inv_dist3 = inv_distance**3

    t = dist2 * (inv_epsilon**2)  # dimensionless ratio

    s_over_dist3 = np.where(t < 0.1, near_limit_constant, inv_dist3)

    dot_n_d = np.sum(n[None, :, :] * d, axis=-1)  # (M, N)
    result = np.sum(dot_n_d * inv_four_pi * s_over_dist3, axis=1)
    return result


def main():
    print("=== Standalone Winder CUDA/DLPack Test ===")

    # 1. Generate Synthetic Input Data
    num_points = 2000
    num_queries = 8000
    radius = 1.0

    pts_np, normals_np = generate_sphere_point_cloud(
        num_points=num_points, radius=radius
    )
    queries_np = np.random.rand(num_queries, 3) * 4 * radius - 2 * radius
    queries_np = queries_np.astype(np.float32)

    beta = 2.0
    epsilon = 250

    print(f"Points: {pts_np.shape}, Normals: {normals_np.shape}")
    print(f"Queries: {queries_np.shape}")

    # 2. Allocate GPU Buffers via CudaBuffer
    pts_buf = CudaBuffer(pts_np.shape, dtype=np.float32)
    normals_buf = CudaBuffer(normals_np.shape, dtype=np.float32)
    queries_buf = CudaBuffer(queries_np.shape, dtype=np.float32)

    out_bf_buf = CudaBuffer((num_queries,), dtype=np.float32)
    out_eng_buf = CudaBuffer((num_queries,), dtype=np.float32)

    # 3. Create CUDA Stream & Transfer Data to GPU
    stream = CudaStream()
    stream_handle = stream.handle

    pts_buf.copy_from_numpy_async(pts_np, stream=stream)
    normals_buf.copy_from_numpy_async(normals_np, stream=stream)
    queries_buf.copy_from_numpy_async(queries_np, stream=stream)

    # -------------------------------------------------------------
    # Test 1: Brute Force Variant
    # -------------------------------------------------------------
    print("\n--- Running Brute Force Variant ---")
    winder.brute_force_winding_numbers(
        pts_buf,
        normals_buf,
        queries_buf,
        out_bf_buf,
        float(epsilon),
        stream_handle,
    )

    out_bf_np = np.zeros(num_queries, dtype=np.float32)
    out_bf_buf.copy_to_numpy_async(out_bf_np, stream=stream)
    stream.synchronize()

    print("Brute Force Results (first 4, last 4):")
    print(out_bf_np[:4], "...", out_bf_np[-4:])

    # -------------------------------------------------------------
    # Test 2: WindingNumbersEngine Variant
    # -------------------------------------------------------------
    print("\n--- Running WindingNumbersEngine Variant ---")
    engine = winder.WindingNumberEngine(pts_buf, normals_buf, stream=stream_handle)
    engine.compute(queries_buf, out_eng_buf, float(beta), float(epsilon), stream_handle)

    out_eng_np = np.zeros(num_queries, dtype=np.float32)
    out_eng_buf.copy_to_numpy_async(out_eng_np, stream=stream)
    stream.synchronize()

    # Force engine destruction while stream is still alive
    del engine
    stream.synchronize()

    print("Engine Results (first 4, last 4):")
    print(out_eng_np[:4], "...", out_eng_np[-4:])

    # -------------------------------------------------------------
    # Reference NumPy computation
    # -------------------------------------------------------------
    print("\n--- NumPy Reference ---")
    ref_np = compute_winding_numbers_numpy(pts_np, normals_np, queries_np, epsilon)
    print("Reference Results (first 4, last 4):")
    print(ref_np[:4], "...", ref_np[-4:])

    # -------------------------------------------------------------
    # Comparison & Validation
    # -------------------------------------------------------------
    max_diff_bf_ref = np.max(np.abs(out_bf_np - ref_np))
    max_diff_eng_ref = np.max(np.abs(out_eng_np - ref_np))
    print(f"\nMax difference (Brute Force vs Reference): {max_diff_bf_ref:.6f}")
    print(f"Max difference (Engine vs Reference):       {max_diff_eng_ref:.6f}")

    if max_diff_bf_ref < 1e-4:
        print("SUCCESS: Brute Force matches reference.")
    else:
        print("WARNING: Brute Force does NOT match reference!")

    if max_diff_eng_ref < 1e-4:
        print("SUCCESS: Engine matches reference.")
    else:
        print("WARNING: Engine does NOT match reference!")


if __name__ == "__main__":
    main()
