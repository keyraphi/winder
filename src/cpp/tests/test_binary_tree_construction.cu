#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <thrust/detail/raw_pointer_cast.h>
#include <vector>

// Include your kernel header
#include "aabb.h"
#include "binary_node.h"
#include "geometry.h"
#include "kernels/build_binary_tree.cuh"
#include "kernels/common.cuh"
#include "tailor_coefficients.h"
#include "vec3.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

class GatherTest : public ::testing::Test {
protected:
  void run_test(const std::vector<float> &h_points_in,
                const std::vector<float> &h_normals_in,
                const std::vector<uint32_t> &h_indices_in) {

    uint32_t count = h_indices_in.size();
    if (count == 0) {
      // Test the kernel with zero count to ensure no-op safety
      interleave_gather_geometry(nullptr, nullptr, nullptr, nullptr, 0);
      return;
    }

    // 1. Transfer to Device using Thrust
    thrust::device_vector<float> d_points = h_points_in;
    thrust::device_vector<float> d_normals = h_normals_in;
    thrust::device_vector<uint32_t> d_indices = h_indices_in;
    thrust::device_vector<float> d_out(count * 6);

    // 2. Launch (Wrapper now handles blocks/threads)
    interleave_gather_geometry(thrust::raw_pointer_cast(d_points.data()),
                               thrust::raw_pointer_cast(d_normals.data()),
                               thrust::raw_pointer_cast(d_indices.data()),
                               thrust::raw_pointer_cast(d_out.data()), count);

    // 3. Transfe
    thrust::host_vector<float> h_out = d_out;

    // 4. Verification
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t src_idx = h_indices_in[i];

      // Expected Point (XYZ)
      EXPECT_FLOAT_EQ(h_out[i * 6 + 0], h_points_in[src_idx * 3 + 0])
          << "Point X mismatch at index " << i;
      EXPECT_FLOAT_EQ(h_out[i * 6 + 1], h_points_in[src_idx * 3 + 1])
          << "Point Y mismatch at index " << i;
      EXPECT_FLOAT_EQ(h_out[i * 6 + 2], h_points_in[src_idx * 3 + 2])
          << "Point Z mismatch at index " << i;

      // Expected Normal (XYZ)
      EXPECT_FLOAT_EQ(h_out[i * 6 + 3], h_normals_in[src_idx * 3 + 0])
          << "Normal X mismatch at index " << i;
      EXPECT_FLOAT_EQ(h_out[i * 6 + 4], h_normals_in[src_idx * 3 + 1])
          << "Normal Y mismatch at index " << i;
      EXPECT_FLOAT_EQ(h_out[i * 6 + 5], h_normals_in[src_idx * 3 + 2])
          << "Normal Z mismatch at index " << i;
    }
  }
};

// --- EDGE CASE: ZERO POINTS ---
TEST_F(GatherTest, HandlesZeroCount) {
  std::vector<float> p = {};
  std::vector<float> n = {};
  std::vector<uint32_t> idx = {};
  // Should not crash or launch invalid memory accesses
  run_test(p, n, idx);
}

// --- TRIVIAL CASE: SEQUENTIAL ---
TEST_F(GatherTest, SequentialGather) {
  std::vector<float> p = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<float> n = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<uint32_t> idx = {0, 1};
  run_test(p, n, idx);
}

// --- STRESS CASE: SCATTERED & LARGE ---
TEST_F(GatherTest, ScatteredGatherLarge) {
  const uint32_t N = 10000;
  std::vector<float> p(N * 3);
  std::vector<float> n(N * 3);
  std::vector<uint32_t> idx(N);

  for (uint32_t i = 0; i < N * 3; ++i) {
    p[i] = (float)i;
    n[i] = (float)i * 0.1f;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), g); // Test uncoalesced access patterns

  run_test(p, n, idx);
}

struct PointToAABB {
  __host__ __device__ AABB operator()(const Vec3 &point) const {
    return point.get_aabb();
  }
};

struct MergeAABB {
  __host__ __device__ AABB operator()(const AABB &a, const AABB &b) const {
    return AABB::merge(a, b);
  }
};

struct MortonTransform {
  Vec3 min_p;
  float scale;
  __host__ __device__ uint32_t operator()(const Vec3 &p) const {
    float tx = (p.x - min_p.x) * scale;
    float ty = (p.y - min_p.y) * scale;
    float tz = (p.z - min_p.z) * scale;
    auto x = static_cast<uint32_t>(fminf(fmaxf(tx * 1024.F, 0.F), 1023.F));
    auto y = static_cast<uint32_t>(fminf(fmaxf(ty * 1024.F, 0.F), 1023.F));
    auto z = static_cast<uint32_t>(fminf(fmaxf(tz * 1024.F, 0.F), 1023.F));
    return morton3D_30bit(x, y, z); // Note: Make sure this helper is visible
  }
};

TEST(MortonPipeline, FullSequenceTest) {
  // 1. Setup Input: 4 points in a clear Z-order
  // In Morton space, (0,0,0) < (0,0,1) < (1,1,1)
  std::vector<Vec3> h_points = {
      {10.0f, 10.0f, 10.0f}, // Point 0: Max corner
      {0.0f, 0.0f, 0.0f},    // Point 1: Min corner
      {5.0f, 5.0f, 5.0f},    // Point 2: Middle
      {0.0f, 0.0f, 1.0f}     // Point 3: Near Min Z-axis
  };
  uint32_t m_count = h_points.size();

  thrust::device_vector<Vec3> d_points = h_points;
  thrust::device_vector<uint32_t> m_morton_codes(m_count);
  thrust::device_vector<uint32_t> m_to_internal(m_count);
  thrust::device_vector<uint32_t> m_to_canonical(m_count);

  auto points_begin = d_points.begin();

  // --- STEP 1: AABB REDUCTION ---
  auto aabb_transform =
      thrust::make_transform_iterator(points_begin, PointToAABB());
  AABB scene_bounds = thrust::reduce(aabb_transform, aabb_transform + m_count,
                                     AABB::empty(), MergeAABB());

  // Verify AABB
  EXPECT_FLOAT_EQ(scene_bounds.min.x, 0.0f);
  EXPECT_FLOAT_EQ(scene_bounds.max.x, 10.0f);

  // --- STEP 2: MORTON GENERATION ---
  Vec3 extent = scene_bounds.max - scene_bounds.min;
  float max_dim = fmaxf(extent.x, fmaxf(extent.y, extent.z));
  float scale = (max_dim > 1e-9f) ? 1.0f / max_dim : 0.0f;
  Vec3 min_p = scene_bounds.min;

  thrust::transform(points_begin, points_begin + m_count,
                    m_morton_codes.begin(), MortonTransform{min_p, scale});

  // --- STEP 3: SORT BY KEY ---
  thrust::sequence(m_to_internal.begin(), m_to_internal.end());
  thrust::sort_by_key(m_morton_codes.begin(), m_morton_codes.end(),
                      m_to_internal.begin());

  // --- STEP 4: SCATTER (Inverse Map) ---
  thrust::scatter(thrust::make_counting_iterator<uint32_t>(0),
                  thrust::make_counting_iterator<uint32_t>(m_count),
                  m_to_internal.begin(), m_to_canonical.begin());

  // --- VERIFICATION ---
  thrust::host_vector<uint32_t> h_to_internal = m_to_internal;
  thrust::host_vector<uint32_t> h_to_canonical = m_to_canonical;

  // Expected order in Morton space:
  // Index 1 (0,0,0) -> Morton 0
  // Index 3 (0,0,1) -> Morton ...
  // Index 2 (5,5,5) -> Morton ...
  // Index 0 (10,10,10) -> Morton Max

  EXPECT_EQ(h_to_internal[0],
            1); // The first internal node should point to original point 1
  EXPECT_EQ(h_to_internal[3],
            0); // The last internal node should point to original point 0

  // Inverse check: Where did point 1 go?
  // Point 1 was (0,0,0), it should be at the start of the sorted list (internal
  // index 0)
  EXPECT_EQ(h_to_canonical[1], 0);
  // Point 0 was (10,10,10), it should be at the end (internal index 3)
  EXPECT_EQ(h_to_canonical[0], 3);
}

TEST(BinaryTopology, BasicHierarchy) {
  // 3 leaves -> 2 internal nodes
  // Morton: 0, 4, 7 (Binary: 000, 100, 111)
  std::vector<uint32_t> h_leaf_codes = {0, 4, 7};
  uint32_t leaf_count = h_leaf_codes.size();
  uint32_t node_count = leaf_count - 1;

  thrust::device_vector<uint32_t> d_codes = h_leaf_codes;
  thrust::device_vector<BinaryNode> d_nodes(node_count);
  thrust::device_vector<uint32_t> d_parents(node_count + leaf_count);

  // Run your kernel wrapper
  build_binary_topology(thrust::raw_pointer_cast(d_codes.data()),
                        thrust::raw_pointer_cast(d_nodes.data()),
                        thrust::raw_pointer_cast(d_parents.data()), leaf_count);

  // Set root parent manually as your code does
  uint32_t root_parent = 0xFFFFFFFF;
  thrust::copy_n(&root_parent, 1, d_parents.begin());

  // Transfer back
  thrust::host_vector<BinaryNode> h_nodes = d_nodes;
  thrust::host_vector<uint32_t> h_parents = d_parents;

  // --- VERIFICATION ---
  // Root (Internal Node 0) should cover [0, 2]
  // Leaves are at indices [leaf_count-1, 2*leaf_count-2]
  // Internal nodes are at [0, leaf_count-2]
  // Since 0 and 4 differ at bit 2, and 4 and 7 differ at bit 1,
  // Node 0 should split at index 0.
  // Left child: Leaf 0. Right child: Internal Node 1 (covering leaves 1 and 2).

  EXPECT_EQ(h_nodes[0].left_child, leaf_count - 1); // 0 + 3 - 1
  EXPECT_TRUE(BinaryNode::is_leaf(h_nodes[0].left_child, leaf_count));

  // Internal Node 1 should have leaves 1 and 2 as children
  EXPECT_EQ(h_nodes[1].left_child, 1 + leaf_count - 1); // 1 + 3 - 1
  EXPECT_TRUE(BinaryNode::is_leaf(h_nodes[1].left_child, leaf_count));
  EXPECT_EQ(h_nodes[1].right_child, 2 + leaf_count - 1); // 2 + 3 - 1
  EXPECT_TRUE(BinaryNode::is_leaf(h_nodes[1].right_child, leaf_count));

  // Parent Check: Leaf 1's [at 3-1+1] parent should be Internal Node 1
  EXPECT_EQ(h_parents[leaf_count - 1 + 1], 1);
  // Roots parent should be 0xFFFFFFFF
  EXPECT_EQ(h_parents[0], 0xFFFFFFFF);
}

TEST(BinaryTopology, DuplicateMortonCodes) {
  // Two leaves with exact same code
  std::vector<uint32_t> h_leaf_codes = {10, 10, 20};
  uint32_t leaf_count = 3;

  thrust::device_vector<uint32_t> d_codes = h_leaf_codes;
  thrust::device_vector<BinaryNode> d_nodes(leaf_count - 1);
  thrust::device_vector<uint32_t> d_parents(leaf_count + (leaf_count - 1));

  build_binary_topology(thrust::raw_pointer_cast(d_codes.data()),
                        thrust::raw_pointer_cast(d_nodes.data()),
                        thrust::raw_pointer_cast(d_parents.data()), leaf_count);

  thrust::host_vector<BinaryNode> h_nodes = d_nodes;

  // For codes [10, 10, 20]:
  // Node 0 splits [0,1] and [2].
  // Left child is Internal Node 1 (index 1). Right child is Leaf 2 (index 2 + 3
  // - 1 = 4).
  EXPECT_EQ(h_nodes[0].left_child, 1);
  EXPECT_FALSE(BinaryNode::is_leaf(h_nodes[0].left_child, leaf_count));
  EXPECT_EQ(h_nodes[0].right_child, 4);
  EXPECT_TRUE(BinaryNode::is_leaf(h_nodes[0].right_child, leaf_count));

  // Node 1 splits [0] and [1] (the two 10s).
  // Left child is Leaf 0 (index 0 + 3 - 1 = 2). Right child is Leaf 1 (index 1
  // + 3 - 1 = 3).
  EXPECT_EQ(h_nodes[1].left_child, 2);
  EXPECT_TRUE(BinaryNode::is_leaf(h_nodes[1].left_child, leaf_count));
  EXPECT_EQ(h_nodes[1].right_child, 3);
  EXPECT_TRUE(BinaryNode::is_leaf(h_nodes[1].right_child, leaf_count));
}

TEST(BinaryTopology, EdgeCaseZeroAndOne) {
  // Case: 0 Leaves
  // Should return immediately without launching or erroring
  build_binary_topology(nullptr, nullptr, nullptr, 0);

  // Case: 1 Leaf
  // Karras algorithm builds L-1 internal nodes. 1-1 = 0 nodes.
  // Should handle gracefully (no-op or just setting parent for the single leaf)
  thrust::device_vector<uint32_t> d_code(1, 123);
  thrust::device_vector<BinaryNode> d_nodes(0); // 0 size
  thrust::device_vector<uint32_t> d_parents(1);

  build_binary_topology(thrust::raw_pointer_cast(d_code.data()),
                        thrust::raw_pointer_cast(d_nodes.data()),
                        thrust::raw_pointer_cast(d_parents.data()), 1);
}

TEST(BinaryTreeRefit, BottomUpAABB) {
  const uint32_t point_count = 64;
  const uint32_t points_per_leaf = 32;
  const uint32_t leaf_count = point_count / points_per_leaf; // 2 leaves
  const uint32_t node_count = leaf_count - 1;                // 1 internal node

  // 1. Create Data: Cluster at -10 and Cluster at +10
  std::vector<PointNormal> h_geometry(point_count); // x,y,z, nx,ny,nz
  for (uint32_t i = 0; i < point_count; ++i) {
    float val = (i < 32) ? -10.0f : 10.0f;
    h_geometry[i].p.x = val;
    h_geometry[i].p.y = val;
    h_geometry[i].p.z = val;
    h_geometry[i].n.x = 0.0f;
    h_geometry[i].n.y = 1.0f;
    h_geometry[i].n.z = 0.0f;
  }

  // 2. Mock Tree Topology:
  // Root (Node 0) has children Leaf 0 and Leaf 1
  // Leaves in your kernel are indexed at (leaf_id + leaf_count - 1)
  // Leaf 0 idx: 2-1+0 = 1 | Leaf 1 idx: 2-1+1 = 2
  std::vector<BinaryNode> h_nodes(node_count);
  h_nodes[0].left_child = 1;
  h_nodes[0].right_child = 2;

  std::vector<uint32_t> h_parents(leaf_count + node_count);
  h_parents[1] = 0;          // Leaf 0's parent is Node 0
  h_parents[2] = 0;          // Leaf 1's parent is Node 0
  h_parents[0] = 0xFFFFFFFF; // Root parent

  // 3. Move to Device
  thrust::device_vector<PointNormal> d_geom = h_geometry;
  thrust::device_vector<BinaryNode> d_nodes = h_nodes;
  thrust::device_vector<uint32_t> d_parents = h_parents;
  thrust::device_vector<AABB> d_aabbs(leaf_count + node_count);
  thrust::device_vector<uint32_t> d_atomic_counters(node_count, 0);
  thrust::device_vector<TailorCoefficientsBf16> d_coefficients(leaf_count);
  // leaf_coefficients would be sized leaf_count

  // 4. Launch
  populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(
      thrust::raw_pointer_cast(d_geom.data()),
      thrust::raw_pointer_cast(d_coefficients.data()), leaf_count,
      thrust::raw_pointer_cast(d_nodes.data()),
      thrust::raw_pointer_cast(d_aabbs.data()),
      thrust::raw_pointer_cast(d_parents.data()),
      thrust::raw_pointer_cast(d_atomic_counters.data()), point_count);

  // 5. Verification
  thrust::host_vector<AABB> h_aabbs = d_aabbs;

  // Check Leaf AABBs
  EXPECT_FLOAT_EQ(h_aabbs[1].min.x, -10.0f);
  EXPECT_FLOAT_EQ(h_aabbs[2].min.x, 10.0f);

  // Check Root AABB (Internal Node 0)
  EXPECT_FLOAT_EQ(h_aabbs[0].min.x, -10.0f);
  EXPECT_FLOAT_EQ(h_aabbs[0].max.x, 10.0f);
}

TEST(BinaryTreeRefit, EdgeCaseZero) {
  // Case: 0 Points / 0 Leaves
  populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(nullptr, nullptr, 0, nullptr,
                                                  nullptr, nullptr, nullptr, 0);
}

TEST(BinaryTreeRefit, SingleLeafCoefficients) {
  const uint32_t point_count = 32;
  const uint32_t leaf_count = 1;

  // 1. Setup Points: 32 points at (1.0, 2.0, 3.0) with normal (0, 0, 1)
  // Geometry layout: x, y, z, nx, ny, nz
  std::vector<PointNormal> h_geometry(point_count);
  Vec3 p_val = {1.0f, 2.0f, 3.0f};
  Vec3 n_val = {0.0f, 0.0f, 1.0f};

  for (uint32_t i = 0; i < point_count; ++i) {
    h_geometry[i].p = p_val;
    h_geometry[i].n = n_val;
  }

  // 2. Mock minimal tree for 1 leaf
  thrust::device_vector<PointNormal> d_geom = h_geometry;
  thrust::device_vector<TailorCoefficientsBf16> d_coeffs(leaf_count);
  thrust::device_vector<AABB> d_aabbs(leaf_count); // Just the leaf AABB
  thrust::device_vector<uint32_t> d_parents(1, 0xFFFFFFFF);
  thrust::device_vector<uint32_t> d_atomic(0); // No internal nodes

  // 3. Launch
  populate_binary_tree_aabb_and_leaf_coefficients<PointNormal>(
      thrust::raw_pointer_cast(d_geom.data()),
      thrust::raw_pointer_cast(d_coeffs.data()), leaf_count,
      nullptr, // No binary nodes needed for 1 leaf
      thrust::raw_pointer_cast(d_aabbs.data()),
      thrust::raw_pointer_cast(d_parents.data()),
      thrust::raw_pointer_cast(d_atomic.data()), point_count);

  // 4. Verify results
  thrust::host_vector<TailorCoefficientsBf16> h_coeffs = d_coeffs;
  thrust::host_vector<AABB> h_aabbs = d_aabbs;

  // Check AABB (Should be a single point AABB)
  EXPECT_FLOAT_EQ(h_aabbs[0].min.x, 1.0f);
  EXPECT_FLOAT_EQ(h_aabbs[0].max.z, 3.0f);

  // Check Zero Order Coefficient
  // If it's a simple sum of normals: 32 * (0, 0, 1) = (0, 0, 32)
  // We convert back to float for comparison
  Vec3 zero_order = Vec3::from_bf16(h_coeffs[0].zero_order);
  EXPECT_NEAR(zero_order.x, 0.0f, 1e-2f);
  EXPECT_NEAR(zero_order.y, 0.0f, 1e-2f);
  EXPECT_NEAR(zero_order.z, 32.0f, 1e-1f);
}


