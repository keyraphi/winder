#include "aabb.h"
#include "binary_node.h"
#include "bvh8.h"
#include "kernels/binary2bvh8.cuh"
#include "kernels/common.cuh"
#include "vec3.h"
#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector>
#include <vector_functions.h>
#include <vector_types.h>

// Helper to simulate quantization for verification
uint8_t host_quantize(float val, float p_min, float p_inv_ext) {
  float quantized = (val - p_min) * p_inv_ext;
  return (uint8_t)std::max(0.0f, std::min(255.0f, std::floor(quantized)));
}

TEST(BVH8Conversion, BalancedCollapse) {
  const uint32_t leaf_count = 8;
  const uint32_t binary_node_count = 7; // L-1

  // 1. Manually construct a balanced binary tree
  // Root (0) -> (1, 2)
  // 1 -> (3, 4), 2 -> (5, 6)
  // 3,4,5,6 are internal nodes splitting into 8 leaves total
  std::vector<BinaryNode> h_bin_nodes(binary_node_count);
  h_bin_nodes[0] = {1, 2};
  h_bin_nodes[1] = {3, 4};
  h_bin_nodes[2] = {5, 6};
  // Indices 7-14 are leaves
  h_bin_nodes[3] = {7, 8};
  h_bin_nodes[4] = {9, 10};
  h_bin_nodes[5] = {11, 12};
  h_bin_nodes[6] = {13, 14};

  // Give them AABBs. Root is largest, children get smaller.
  std::vector<AABB> h_bin_aabbs(binary_node_count + leaf_count);
  for (uint32_t i = 0; i < h_bin_aabbs.size(); ++i) {
    h_bin_aabbs[i].min = Vec3{(float)-i, (float)-i, (float)-i};
    h_bin_aabbs[i].max = Vec3{(float)i, (float)i, (float)i};
  }

  // 2. Setup BVH8 Params
  thrust::device_vector<uint32_t> d_qA(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_qB(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_int_parents(leaf_count, 0);
  thrust::device_vector<uint32_t> d_counter(1, 0);
  thrust::device_vector<BinaryNode> d_bin_nodes = h_bin_nodes;
  thrust::device_vector<AABB> d_bin_aabbs = h_bin_aabbs;
  thrust::device_vector<uint32_t> d_leaf_parents(leaf_count, 0);
  thrust::device_vector<BVH8Node> d_bvh8_nodes(leaf_count);
  thrust::device_vector<LeafPointers> d_leaf_pointers(leaf_count);

  ConvertBinary2BVH8Params p{thrust::raw_pointer_cast(d_qA.data()),
                             thrust::raw_pointer_cast(d_qB.data()),
                             thrust::raw_pointer_cast(d_int_parents.data()),
                             thrust::raw_pointer_cast(d_counter.data()),
                             leaf_count,
                             thrust::raw_pointer_cast(d_bin_aabbs.data()),
                             thrust::raw_pointer_cast(d_bin_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_parents.data()),
                             thrust::raw_pointer_cast(d_bvh8_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_pointers.data())};
  // 3. Launch (Note: Requires Cooperative Launch support on GPU)
  convert_binary_tree_to_bvh8(p, 0);
  // 4. Detailed Verification
  thrust::host_vector<BVH8Node> h_bvh8 = d_bvh8_nodes;
  thrust::host_vector<LeafPointers> h_leaf_ptrs = d_leaf_pointers;

  // Check Root Node (Index 0)
  BVH8Node root = h_bvh8[0];

  // A. Verify Parent AABB
  EXPECT_FLOAT_EQ(root.parent_aabb.min.x, h_bin_aabbs[0].min.x);
  EXPECT_FLOAT_EQ(root.parent_aabb.max.z, h_bin_aabbs[0].max.z);

  // Verify Child Meta and Leaf Pointers (Order Independent)
  std::vector<uint32_t> found_indices;
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(root.getChildMeta(i), ChildType::LEAF);
    found_indices.push_back(h_leaf_ptrs[0].indices[i]);
  }

  // Sort and compare against expected set {0, 1, 2, 3, 4, 5, 6, 7}
  std::sort(found_indices.begin(), found_indices.end());
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(found_indices[i], i) << "Missing raw leaf index " << i;
  }

  // C. Verify AABB Quantization (Checking all children since we don't know the
  // order)
  Vec3 p_min = root.parent_aabb.min;
  Vec3 p_inv_ext = 255.0f / (root.parent_aabb.max - root.parent_aabb.min);

  for (uint32_t i = 0; i < 8; ++i) {
    uint32_t leaf_raw_idx = h_leaf_ptrs[0].indices[i];
    uint32_t binary_idx = leaf_raw_idx + (leaf_count - 1);
    AABB child_box = h_bin_aabbs[binary_idx];

    EXPECT_EQ(root.child_aabb_approx[i].qmin[0],
              host_quantize(child_box.min.x, p_min.x, p_inv_ext.x));
    EXPECT_EQ(root.child_aabb_approx[i].qmax[0],
              host_quantize(child_box.max.x, p_min.x, p_inv_ext.x));
  }

  // D. Verify Child Base
  EXPECT_EQ(root.child_base, 1);
}

TEST(BVH8Conversion, UniformLinearLeaves) {
  const uint32_t leaf_count = 100;
  const uint32_t internal_count = leaf_count - 1;
  std::vector<BinaryNode> h_bin_nodes(internal_count);
  std::vector<AABB> h_bin_aabbs(internal_count + leaf_count);

  // Build a simple "Right-leaning" chain
  // Node i -> Left: Leaf i, Right: Node i+1 (or Leaf 99 for the last one)
  for (uint32_t i = 0; i < internal_count; ++i) {
    h_bin_nodes[i].left_child = internal_count + i; // Leaf i
    if (i < internal_count - 1) {
      h_bin_nodes[i].right_child = i + 1; // Next Internal
    } else {
      h_bin_nodes[i].right_child = internal_count + i + 1; // Last Leaf
    }
  }

  // AABBs: Each leaf is [i, i+1]. Internal node i contains leaves [i, 99]
  for (uint32_t i = 0; i < leaf_count; ++i) {
    h_bin_aabbs[internal_count + i].min = Vec3{(float)i, 0.0f, 0.0f};
    h_bin_aabbs[internal_count + i].max = Vec3{(float)i + 1.0f, 1.0f, 1.0f};
  }
  // Bottom-up manual refit for internal nodes
  for (int i = internal_count - 1; i >= 0; --i) {
    h_bin_aabbs[i] = AABB::merge(h_bin_aabbs[h_bin_nodes[i].left_child],
                               h_bin_aabbs[h_bin_nodes[i].right_child]);
  }

  // Setup BVH8 Params
  thrust::device_vector<uint32_t> d_qA(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_qB(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_int_parents(leaf_count, 0);
  thrust::device_vector<uint32_t> d_counter(1, 0);
  thrust::device_vector<BinaryNode> d_bin_nodes = h_bin_nodes;
  thrust::device_vector<AABB> d_bin_aabbs = h_bin_aabbs;
  thrust::device_vector<uint32_t> d_leaf_parents(leaf_count, 0);
  thrust::device_vector<BVH8Node> d_bvh8_nodes(leaf_count);
  thrust::device_vector<LeafPointers> d_leaf_pointers(leaf_count);

  ConvertBinary2BVH8Params p{thrust::raw_pointer_cast(d_qA.data()),
                             thrust::raw_pointer_cast(d_qB.data()),
                             thrust::raw_pointer_cast(d_int_parents.data()),
                             thrust::raw_pointer_cast(d_counter.data()),
                             leaf_count,
                             thrust::raw_pointer_cast(d_bin_aabbs.data()),
                             thrust::raw_pointer_cast(d_bin_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_parents.data()),
                             thrust::raw_pointer_cast(d_bvh8_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_pointers.data())};
  // Launch (Note: Requires Cooperative Launch support on GPU)
  convert_binary_tree_to_bvh8(p, 0);
  // Detailed Verification
  thrust::host_vector<BVH8Node> h_bvh8 = d_bvh8_nodes;

  // Verification: Root (0) must have split until it has 8 children.
  // In a chain, it splits nodes 0, 1, 2, 3, 4, 5, 6.
  // Children should be Leaves 0-6 and Internal Node 7.
  EXPECT_EQ(h_bvh8[0].child_base, 1);
  uint32_t leaves_found = 0;
  uint32_t internal_found = 0;
  for (int i = 0; i < 8; ++i) {
    if (h_bvh8[0].getChildMeta(i) == ChildType::LEAF)
      leaves_found++;
    if (h_bvh8[0].getChildMeta(i) == ChildType::INTERNAL)
      internal_found++;
  }
  EXPECT_EQ(leaves_found, 7);
  EXPECT_EQ(internal_found, 1);

  thrust::host_vector<uint32_t> h_leaf_parents = d_leaf_parents;
  thrust::host_vector<LeafPointers> h_leaf_ptrs = d_leaf_pointers;

  // 1. Verify Topology
  EXPECT_EQ(h_bvh8[0].child_base, 1);
  EXPECT_EQ(h_bvh8[0].getChildMeta(7), ChildType::INTERNAL);

  // 2. Verify Leaf-to-Parent Mapping
  // In a chain, Leaf 0-6 should be children of BVH8 Node 0 (Root)
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(h_leaf_parents[i], 0) << "Leaf " << i << " should point to Root";
  }

  // 3. Verify Parent-to-Leaf Mapping (LeafPointers)
  // LeafPointers for Node 0 should contain indices 0 through 6
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(h_leaf_ptrs[0].indices[i], i);
  }
  // The 8th slot (index 7) of Node 0 is INTERNAL, so indices[7] should be
  // 0xFFFFFFFF
  EXPECT_EQ(h_leaf_ptrs[0].indices[7], 0xFFFFFFFF);

  // 4. Verify AABB Chain Logic
  // The root AABB must encompass the entire range [0, 100]
  EXPECT_FLOAT_EQ(h_bvh8[0].parent_aabb.min.x, 0.0f);
  EXPECT_FLOAT_EQ(h_bvh8[0].parent_aabb.max.x, 100.0f);
}

TEST(BVH8Conversion, NonUniformHeuristic) {
  const uint32_t leaf_count = 16;
  const uint32_t internal_count = leaf_count - 1;
  std::vector<BinaryNode> h_bin_nodes(internal_count);
  std::vector<AABB> h_bin_aabbs(internal_count + leaf_count);

  // Node 0 -> Left: Leaf 0, Right: Node 1
  // Node 1 -> Left: Leaf 1, Right: Node 2...
  for (uint32_t i = 0; i < internal_count; ++i) {
    h_bin_nodes[i].left_child = internal_count + i;
    h_bin_nodes[i].right_child =
        (i < internal_count - 1) ? (i + 1) : (internal_count + i + 1);
  }

  // Leaves: Leaf i has size i*i (exponential growth)
  for (uint32_t i = 0; i < leaf_count; ++i) {
    float size = (float)(i + 1);
    h_bin_aabbs[internal_count + i].min = Vec3{0, 0, 0};
    h_bin_aabbs[internal_count + i].max = Vec3{size, size, size};
  }
  for (int i = internal_count - 1; i >= 0; --i) {
    h_bin_aabbs[i] = AABB::merge(h_bin_aabbs[h_bin_nodes[i].left_child],
                                 h_bin_aabbs[h_bin_nodes[i].right_child]);
  }

  // Setup BVH8 Params
  thrust::device_vector<uint32_t> d_qA(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_qB(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_int_parents(leaf_count, 0);
  thrust::device_vector<uint32_t> d_counter(1, 0);
  thrust::device_vector<BinaryNode> d_bin_nodes = h_bin_nodes;
  thrust::device_vector<AABB> d_bin_aabbs = h_bin_aabbs;
  thrust::device_vector<uint32_t> d_leaf_parents(leaf_count, 0);
  thrust::device_vector<BVH8Node> d_bvh8_nodes(leaf_count);
  thrust::device_vector<LeafPointers> d_leaf_pointers(leaf_count);

  ConvertBinary2BVH8Params p{thrust::raw_pointer_cast(d_qA.data()),
                             thrust::raw_pointer_cast(d_qB.data()),
                             thrust::raw_pointer_cast(d_int_parents.data()),
                             thrust::raw_pointer_cast(d_counter.data()),
                             leaf_count,
                             thrust::raw_pointer_cast(d_bin_aabbs.data()),
                             thrust::raw_pointer_cast(d_bin_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_parents.data()),
                             thrust::raw_pointer_cast(d_bvh8_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_pointers.data())};
  // Launch (Note: Requires Cooperative Launch support on GPU)
  convert_binary_tree_to_bvh8(p, 0);
  // Detailed Verification
  thrust::host_vector<BVH8Node> h_bvh8 = d_bvh8_nodes;
  thrust::host_vector<LeafPointers> h_leaf_ptrs = d_leaf_pointers;

  // Verification: Because Node 1 is much larger than Leaf 0,
  // the root should immediately split Node 1, then Node 2, etc.
  // This ensures child_meta isn't just filled sequentially.
  EXPECT_GT(h_bvh8[0].parent_aabb.max.x, 0.0f);

  BVH8Node root = h_bvh8[0];
  Vec3 p_min = root.parent_aabb.min;
  Vec3 p_max = root.parent_aabb.max;
  Vec3 p_inv_ext = Vec3{255.0f, 255.0f, 255.0f} / (p_max - p_min);

  // Verify the quantization of Leaf 0 within the Root Node
  // Leaf 0 is at [0, 1], Root is at [0, 16] (approx, based on your merge)
  for (int i = 0; i < 8; ++i) {
    if (root.getChildMeta(i) == ChildType::LEAF) {
      uint32_t leaf_idx = h_leaf_ptrs[0].indices[i];
      AABB original_leaf_box = h_bin_aabbs[internal_count + leaf_idx];

      // Manual host-side quantization to compare
      uint8_t expected_qmin_x = (uint8_t)std::round(
          (original_leaf_box.min.x - p_min.x) * p_inv_ext.x);
      uint8_t expected_qmax_x = (uint8_t)std::round(
          (original_leaf_box.max.x - p_min.x) * p_inv_ext.x);

      EXPECT_EQ(root.child_aabb_approx[i].qmin[0], expected_qmin_x);
      EXPECT_EQ(root.child_aabb_approx[i].qmax[0], expected_qmax_x);
    }
  }
}

TEST(BVH8Conversion, DegeneratePointCluster64) {
  const uint32_t leaf_count = 64;
  const uint32_t internal_count = leaf_count - 1;
  std::vector<BinaryNode> h_bin_nodes(internal_count);
  std::vector<AABB> h_bin_aabbs(internal_count + leaf_count);

  // 1. Force the heuristic: Give higher nodes larger AABBs
  for (uint32_t i = 0; i < internal_count + leaf_count; ++i) {
    float size = (float)(internal_count + leaf_count - i);
    h_bin_aabbs[i].min = Vec3{0, 0, 0};
    h_bin_aabbs[i].max = Vec3{size, size, size};
  }

  // 2. Build the binary tree
  for (uint32_t i = 0; i < internal_count; ++i) {
    h_bin_nodes[i].left_child = 2 * i + 1;
    h_bin_nodes[i].right_child = 2 * i + 2;
  }

  // Setup BVH8 Params
  thrust::device_vector<uint32_t> d_qA(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_qB(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_int_parents(leaf_count, 0);
  thrust::device_vector<uint32_t> d_counter(1, 0);
  thrust::device_vector<BinaryNode> d_bin_nodes = h_bin_nodes;
  thrust::device_vector<AABB> d_bin_aabbs = h_bin_aabbs;
  thrust::device_vector<uint32_t> d_leaf_parents(leaf_count, 0);
  thrust::device_vector<BVH8Node> d_bvh8_nodes(leaf_count);
  thrust::device_vector<LeafPointers> d_leaf_pointers(leaf_count);

  ConvertBinary2BVH8Params p{thrust::raw_pointer_cast(d_qA.data()),
                             thrust::raw_pointer_cast(d_qB.data()),
                             thrust::raw_pointer_cast(d_int_parents.data()),
                             thrust::raw_pointer_cast(d_counter.data()),
                             leaf_count,
                             thrust::raw_pointer_cast(d_bin_aabbs.data()),
                             thrust::raw_pointer_cast(d_bin_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_parents.data()),
                             thrust::raw_pointer_cast(d_bvh8_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_pointers.data())};
  // Launch (Note: Requires Cooperative Launch support on GPU)
  convert_binary_tree_to_bvh8(p, 0);
  thrust::host_vector<BVH8Node> h_bvh8 = d_bvh8_nodes;
  thrust::host_vector<uint32_t> h_leaf_parents = d_leaf_parents;

  // 3. Verification
  // The Root (0) should have 8 internal children (Nodes 1-8)
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(h_bvh8[0].getChildMeta(i), ChildType::INTERNAL);
  }
  EXPECT_EQ(h_bvh8[0].child_base, 1);

  // 4. Level 1 (Nodes 1-8) should contain ONLY leaves
  for (int n = 1; n <= 8; ++n) {
    for (int c = 0; c < 8; ++c) {
      EXPECT_EQ(h_bvh8[n].getChildMeta(c), ChildType::LEAF);
    }
    // Since nodes 1-8 have NO internal children, their child_base
    // points to the start of the (non-existent) Level 2.
    // Level 2 start = Level 1 Start (1) + Level 1 Width (8) = 9.
    EXPECT_EQ(h_bvh8[n].child_base, 9);
  }

  // 5. Leaf Parent Verification
  // Leaf 63 (Binary 63 + 63 = 126) should be in the last node (8)
  EXPECT_EQ(h_leaf_parents[63], 8);
}

TEST(BVH8Conversion, SingleLeafBaseCase) {
  const uint32_t leaf_count = 1;
  std::vector<BinaryNode> h_bin_nodes(0); // No internal nodes
  std::vector<AABB> h_bin_aabbs(1);
  AABB h_aabb;
  h_aabb.min = Vec3{1, 1, 1};
  h_aabb.max = Vec3{2, 2, 2};
  h_bin_aabbs[0] = h_aabb;

  // Setup BVH8 Params
  thrust::device_vector<uint32_t> d_qA(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_qB(leaf_count * 2, 0);
  thrust::device_vector<uint32_t> d_int_parents(leaf_count, 0);
  thrust::device_vector<uint32_t> d_counter(1, 0);
  thrust::device_vector<BinaryNode> d_bin_nodes = h_bin_nodes;
  thrust::device_vector<AABB> d_bin_aabbs = h_bin_aabbs;
  thrust::device_vector<uint32_t> d_leaf_parents(leaf_count, 0);
  thrust::device_vector<BVH8Node> d_bvh8_nodes(leaf_count);
  thrust::device_vector<LeafPointers> d_leaf_pointers(leaf_count);

  ConvertBinary2BVH8Params p{thrust::raw_pointer_cast(d_qA.data()),
                             thrust::raw_pointer_cast(d_qB.data()),
                             thrust::raw_pointer_cast(d_int_parents.data()),
                             thrust::raw_pointer_cast(d_counter.data()),
                             leaf_count,
                             thrust::raw_pointer_cast(d_bin_aabbs.data()),
                             thrust::raw_pointer_cast(d_bin_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_parents.data()),
                             thrust::raw_pointer_cast(d_bvh8_nodes.data()),
                             thrust::raw_pointer_cast(d_leaf_pointers.data())};
  // Launch (Note: Requires Cooperative Launch support on GPU)
  convert_binary_tree_to_bvh8(p, 0);
  thrust::host_vector<BVH8Node> h_bvh8 = d_bvh8_nodes;
  thrust::host_vector<LeafPointers> h_leaf_ptrs = d_leaf_pointers;

  // Verification
  EXPECT_EQ(h_bvh8[0].getChildMeta(0), ChildType::LEAF);
  EXPECT_EQ(h_bvh8[0].getChildMeta(1), ChildType::EMPTY);
  EXPECT_EQ(h_bvh8[0].parent_aabb.min.x, 1.0f);

  BVH8Node root = h_bvh8[0];
  LeafPointers root_leafs = h_leaf_ptrs[0];

  // Verify active slot
  EXPECT_EQ(root.getChildMeta(0), ChildType::LEAF);
  EXPECT_EQ(root_leafs.indices[0], 0);

  // Verify all 7 other slots are strictly neutralized
  for (int i = 1; i < 8; ++i) {
    EXPECT_EQ(root.getChildMeta(i), ChildType::EMPTY)
        << "Slot " << i << " should be EMPTY";
    EXPECT_EQ(root_leafs.indices[i], 0xFFFFFFFF)
        << "Slot " << i << " leaf pointer should be null";
  }
}
