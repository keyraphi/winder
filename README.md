# winder

Building massively parallel, GPU-accelerated C++/CUDA spatial structures to evaluate geometric winding numbers with extreme performance.

> ⚠️ **Active Beta Warning**
> `winder` is currently in active pre-release development. The API is unstable and subject to breaking changes. Advanced features—such as analytical field derivatives—are currently being implemented. Use with caution in production environments.

---

## 💡 The Concept

`winder` is designed to compute 3D geometric winding numbers across massive query grids entirely on the GPU. By leveraging highly optimized spatial data structures (such as specialized 8-ary Bounding Volume Hierarchies), the library collapses the typical $\mathcal{O}(N \times M)$ brute-force complexity of mesh evaluation down to high-speed logarithmic bounds. 

Whether you are evaluating solid angles of open sheets, identifying inside/outside boundaries of complex closed manifolds, or computing continuous volumetric fields for graphics and physical simulations, `winder` handles millions of primitives and queries seamlessly. Integration to torch, tensorflow, jax, etc works via DLPack-capsules.

For an in-depth dive into the underlying mathematical principles, spatial layout partitioning, and performance benchmarks, read our formal write-up:

👉 **[Read the Technical Report](https://keyraphi.github.io/winder/)**

---

## 🚀 Installation

Because `winder` is currently in beta, releases are published to PyPI as pre-releases. To install it, you must explicitly pass the `--pre` flag to your package manager:

```bash
pip install winder --pre
'''

## System Requirements

* OS: Linux (ManyLinux 2.28+ compliant distributions)
* Compute: NVIDIA GPU
* Driver: NVIDIA Driver supporting CUDA 13.0 or newer (the wheel bundles its own CUDA 13 runtime components, but relies on your host system driver).

## 🛠️ Quick Start & Usage

winder integrates directly with PyTorch tensors, executing heavy compute pipelines in CUDA space without unnecessary host-to-device memory roundtrips.

1. Interactive Documentation
You can explore the available internal module bindings, arguments, and function structures directly through Python's built-in inspection tool:

```python


import winder
# View the full module documentation, signatures, and docstrings
help(winder)
```

2. Basic Example: Brute-Force vs. Accelerated Compute

Here is a minimal script generating spatial queries and evaluating their winding numbers using both the exact baseline evaluator and the optimized hierarchical tree system:

```python

import torch
import winder

# Verify CUDA availability
if not torch.cuda.is_available():
    print("No cuda device found. Returning.")
    exit()

device = torch.device("cuda:0")
print(f"Running winder on device: {device}")

#  Generate a batch of spatial query points (e.g., 1000,000 points in 3D space)
queries = torch.rand((1000000, 3), dtype=torch.float32, device=device)

# Define a simple triangle mesh layout [num_triangles, 3_vertices, 3_coordinates]
# Here we define a single right-angle triangle positioned on the XY plane
triangles = torch.tensor([
    [[0.0, 0.0, 0.0], 
     [1.0, 0.0, 0.0], 
     [0.0, 1.0, 0.0]]
], dtype=torch.float32, device=device)

# Create winder enginge
engine = winder.WinderEngine(triangles)

print("Evaluating winding fields...")
# Method A: Calculate using the exact O(N*M) brute force method
winding_brute = engine.brute_force(queries)

# Method B: Calculate using the optimized accelerated tree structure
winding_fast = engine.compute(queries)

# Load DLPack-capsules into torch tensors
winding_brute = torch.from_dlpack(winding_brute)
winding_fast = torch.from_dlpack(winding_fast)

# Compare results
print(f"Brute-force output shape: {winding_brute.shape}")
print(f"Accelerated output shape: {winding_fast.shape}")
# ... do whatever you want
```

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
