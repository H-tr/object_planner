#pragma once

#include "data_structures.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include <xsimd/xsimd.hpp>

namespace object_planner {

// Per-leaf storage for the obstacle BSP tree. Obstacle points live in a
// structure-of-arrays layout (one xsimd-aligned float vector per axis)
// padded with sentinel coordinates so the SIMD inner loop can do
// aligned loads without a per-iteration tail check.
struct ObstacleLeaf {
  using AlignedVector = std::vector<float, xsimd::aligned_allocator<float>>;
  AlignedVector xs, ys, zs;
  std::uint32_t count = 0; // unpadded point count (xs.size() may be larger)
};

// A node in the obstacle BSP tree. `axis == -1` marks a leaf, in which
// case `leaf_index` indexes into the checker's `leaves_` vector.
struct BspNode {
  std::array<float, 6> aabb_padded{}; // {x_min, x_max, y_min, y_max, z_min, z_max}
                                      // pre-inflated by point_inflation
  int axis = -1;                      // 0/1/2 for inner nodes, -1 for leaves
  float split = 0.0f;                 // split coordinate on `axis`
  int left = -1;
  int right = -1;
  int leaf_index = -1;
};

// SoA view of the object's leaf spheres. Used so future versions can
// also SIMD the per-config transform; today the transform stays scalar
// while the obstacle-side per-leaf check is the SIMD hot loop.
struct ObjectLeafSpheresSoA {
  using AlignedVector = std::vector<float, xsimd::aligned_allocator<float>>;
  AlignedVector center_x, center_y, center_z;
  AlignedVector radius;
  std::size_t count = 0;
};

// Collision checker: object sphere tree (only leaves are used) against
// an obstacle point cloud where every point is treated as a sphere of
// radius ``point_inflation``. The obstacle side is stored in an
// axis-alternating BSP tree with SoA leaves; the inner distance test
// is genuinely SIMD via xsimd. The object side is transformed by the
// SE(2) config and queried scalar against the tree (the tree's
// per-leaf check is the SIMD-batched hot loop).
class BatchedCollisionChecker {
public:
  inline BatchedCollisionChecker(
      const std::vector<SphereTreeNode> &object_sphere_tree,
      const std::vector<Point3D> &obstacle_points, float point_inflation);

  // Returns true as soon as any config in `path_segment` collides.
  inline bool
  is_path_in_collision(const std::vector<Config> &path_segment) const;

private:
  static constexpr std::size_t kLeafTarget = 32;
  using FBatch = xsimd::batch<float>;
  static constexpr std::size_t kBatchSize = FBatch::size;

  inline void build_object_soa(const std::vector<SphereTreeNode> &tree);
  inline void build_bsp(const std::vector<Point3D> &obstacle_points);
  inline int build_bsp_recursive(std::vector<int> &indices,
                                  const std::vector<Point3D> &points);
  inline void emit_leaf(int node_id, const std::vector<int> &indices,
                         const std::vector<Point3D> &points);
  inline bool query_sphere(float cx, float cy, float cz, float r_query) const;
  inline bool query_node(int node_id, float cx, float cy, float cz,
                          float r_query, float threshold_sq) const;

  static inline std::size_t round_up_to_batch(std::size_t n) {
    const std::size_t rem = n % kBatchSize;
    return rem == 0 ? n : n + (kBatchSize - rem);
  }

  static inline bool aabb_sphere_intersect(const std::array<float, 6> &aabb,
                                             float cx, float cy, float cz,
                                             float r) {
    float dx = 0.0f;
    if (cx < aabb[0]) dx = aabb[0] - cx;
    else if (cx > aabb[1]) dx = cx - aabb[1];
    float dy = 0.0f;
    if (cy < aabb[2]) dy = aabb[2] - cy;
    else if (cy > aabb[3]) dy = cy - aabb[3];
    float dz = 0.0f;
    if (cz < aabb[4]) dz = aabb[4] - cz;
    else if (cz > aabb[5]) dz = cz - aabb[5];
    return dx * dx + dy * dy + dz * dz <= r * r;
  }

  ObjectLeafSpheresSoA object_soa_;
  std::vector<BspNode> nodes_;
  std::vector<ObstacleLeaf> leaves_;
  float point_inflation_ = 0.0f;
  int root_id_ = -1;
};

inline BatchedCollisionChecker::BatchedCollisionChecker(
    const std::vector<SphereTreeNode> &object_sphere_tree,
    const std::vector<Point3D> &obstacle_points, float point_inflation)
    : point_inflation_(point_inflation < 0.0f ? 0.0f : point_inflation) {
  build_object_soa(object_sphere_tree);
  build_bsp(obstacle_points);
}

inline void BatchedCollisionChecker::build_object_soa(
    const std::vector<SphereTreeNode> &tree) {
  for (const auto &node : tree) {
    if (node.child1_id == -1) { // leaf
      object_soa_.center_x.push_back(static_cast<float>(node.sphere.center.x()));
      object_soa_.center_y.push_back(static_cast<float>(node.sphere.center.y()));
      object_soa_.center_z.push_back(static_cast<float>(node.sphere.center.z()));
      object_soa_.radius.push_back(static_cast<float>(node.sphere.radius));
    }
  }
  object_soa_.count = object_soa_.center_x.size();
  const std::size_t padded = round_up_to_batch(object_soa_.count);
  object_soa_.center_x.resize(padded, 0.0f);
  object_soa_.center_y.resize(padded, 0.0f);
  object_soa_.center_z.resize(padded, 0.0f);
  object_soa_.radius.resize(padded, 0.0f);
}

inline void BatchedCollisionChecker::build_bsp(
    const std::vector<Point3D> &obstacle_points) {
  if (obstacle_points.empty()) {
    return;
  }
  std::vector<int> indices(obstacle_points.size());
  std::iota(indices.begin(), indices.end(), 0);
  root_id_ = build_bsp_recursive(indices, obstacle_points);
}

inline void BatchedCollisionChecker::emit_leaf(
    int node_id, const std::vector<int> &indices,
    const std::vector<Point3D> &points) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  ObstacleLeaf leaf;
  const std::size_t padded = round_up_to_batch(indices.size());
  leaf.xs.resize(padded, kInf);
  leaf.ys.resize(padded, kInf);
  leaf.zs.resize(padded, kInf);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const auto &p = points[indices[i]];
    leaf.xs[i] = static_cast<float>(p.x());
    leaf.ys[i] = static_cast<float>(p.y());
    leaf.zs[i] = static_cast<float>(p.z());
  }
  leaf.count = static_cast<std::uint32_t>(indices.size());
  nodes_[node_id].leaf_index = static_cast<int>(leaves_.size());
  leaves_.push_back(std::move(leaf));
}

inline int BatchedCollisionChecker::build_bsp_recursive(
    std::vector<int> &indices, const std::vector<Point3D> &points) {
  if (indices.empty()) {
    return -1;
  }

  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::array<float, 6> aabb{kInf, -kInf, kInf, -kInf, kInf, -kInf};
  for (int idx : indices) {
    const auto &p = points[idx];
    aabb[0] = std::min(aabb[0], static_cast<float>(p.x()));
    aabb[1] = std::max(aabb[1], static_cast<float>(p.x()));
    aabb[2] = std::min(aabb[2], static_cast<float>(p.y()));
    aabb[3] = std::max(aabb[3], static_cast<float>(p.y()));
    aabb[4] = std::min(aabb[4], static_cast<float>(p.z()));
    aabb[5] = std::max(aabb[5], static_cast<float>(p.z()));
  }
  // Pre-inflate by point_inflation so query-time tests use the raw query
  // sphere radius (no per-traversal inflation).
  aabb[0] -= point_inflation_;
  aabb[1] += point_inflation_;
  aabb[2] -= point_inflation_;
  aabb[3] += point_inflation_;
  aabb[4] -= point_inflation_;
  aabb[5] += point_inflation_;

  const int node_id = static_cast<int>(nodes_.size());
  nodes_.emplace_back();
  nodes_[node_id].aabb_padded = aabb;

  if (indices.size() <= kLeafTarget) {
    emit_leaf(node_id, indices, points);
    return node_id;
  }

  // Choose the widest unpadded axis.
  const float ext_x = (aabb[1] - point_inflation_) - (aabb[0] + point_inflation_);
  const float ext_y = (aabb[3] - point_inflation_) - (aabb[2] + point_inflation_);
  const float ext_z = (aabb[5] - point_inflation_) - (aabb[4] + point_inflation_);
  int axis = 0;
  float ext = ext_x;
  if (ext_y > ext) { axis = 1; ext = ext_y; }
  if (ext_z > ext) { axis = 2; ext = ext_z; }
  if (ext <= 0.0f) {
    // Degenerate: every point coincides on the longest axis. Force leaf.
    emit_leaf(node_id, indices, points);
    return node_id;
  }

  const std::size_t mid = indices.size() / 2;
  std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
                   [&](int a, int b) {
                     return points[a][axis] < points[b][axis];
                   });
  const float split = static_cast<float>(points[indices[mid]][axis]);
  std::vector<int> left_idx(indices.begin(), indices.begin() + mid);
  std::vector<int> right_idx(indices.begin() + mid, indices.end());

  nodes_[node_id].axis = axis;
  nodes_[node_id].split = split;
  const int left = build_bsp_recursive(left_idx, points);
  const int right = build_bsp_recursive(right_idx, points);
  nodes_[node_id].left = left;
  nodes_[node_id].right = right;
  return node_id;
}

inline bool BatchedCollisionChecker::query_sphere(float cx, float cy, float cz,
                                                   float r_query) const {
  if (root_id_ < 0) return false;
  const float r_combined = r_query + point_inflation_;
  const float threshold_sq = r_combined * r_combined;
  return query_node(root_id_, cx, cy, cz, r_query, threshold_sq);
}

inline bool BatchedCollisionChecker::query_node(int node_id, float cx, float cy,
                                                 float cz, float r_query,
                                                 float threshold_sq) const {
  const BspNode &node = nodes_[node_id];
  if (!aabb_sphere_intersect(node.aabb_padded, cx, cy, cz, r_query)) {
    return false;
  }
  if (node.axis == -1) {
    const ObstacleLeaf &leaf = leaves_[node.leaf_index];
    auto cx_b = FBatch::broadcast(cx);
    auto cy_b = FBatch::broadcast(cy);
    auto cz_b = FBatch::broadcast(cz);
    auto thr_b = FBatch::broadcast(threshold_sq);
    const std::size_t padded = leaf.xs.size();
    for (std::size_t i = 0; i < padded; i += kBatchSize) {
      auto xs = FBatch::load_aligned(&leaf.xs[i]);
      auto ys = FBatch::load_aligned(&leaf.ys[i]);
      auto zs = FBatch::load_aligned(&leaf.zs[i]);
      auto dx = xs - cx_b;
      auto dy = ys - cy_b;
      auto dz = zs - cz_b;
      auto distsq = dx * dx + dy * dy + dz * dz;
      auto mask = distsq <= thr_b;
      if (xsimd::any(mask)) return true;
    }
    return false;
  }
  // Inner: descend near child first.
  const float coord = (node.axis == 0) ? cx : (node.axis == 1) ? cy : cz;
  const int near_child = (coord < node.split) ? node.left : node.right;
  const int far_child = (coord < node.split) ? node.right : node.left;
  if (near_child >= 0 &&
      query_node(near_child, cx, cy, cz, r_query, threshold_sq)) {
    return true;
  }
  if (far_child >= 0 &&
      query_node(far_child, cx, cy, cz, r_query, threshold_sq)) {
    return true;
  }
  return false;
}

inline bool BatchedCollisionChecker::is_path_in_collision(
    const std::vector<Config> &path_segment) const {
  if (object_soa_.count == 0 || root_id_ < 0) {
    return false;
  }
  for (const auto &config : path_segment) {
    const float c = std::cos(static_cast<float>(config.theta));
    const float s = std::sin(static_cast<float>(config.theta));
    const float tx = static_cast<float>(config.x);
    const float ty = static_cast<float>(config.y);
    for (std::size_t i = 0; i < object_soa_.count; ++i) {
      const float ox = object_soa_.center_x[i];
      const float oy = object_soa_.center_y[i];
      const float oz = object_soa_.center_z[i];
      const float r = object_soa_.radius[i];
      const float wx = c * ox - s * oy + tx;
      const float wy = s * ox + c * oy + ty;
      const float wz = oz;
      if (query_sphere(wx, wy, wz, r)) {
        return true;
      }
    }
  }
  return false;
}

} // namespace object_planner
