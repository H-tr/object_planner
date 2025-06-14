#pragma once

#include "data_structures.h"
#include "nanoflann.hpp"
#include <memory>
#include <vector>
#include <xsimd/xsimd.hpp>

namespace object_planner {

// Adapter for nanoflann KD-tree
struct PointCloud {
  std::vector<Point3D> pts;
  inline size_t kdtree_get_point_count() const { return pts.size(); }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return pts[idx][dim];
  }
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 3>;

// SoA representation of the object's leaf spheres for SIMD
struct ObjectLeafSpheresSoA {
  using AlignedVector = std::vector<double, xsimd::aligned_allocator<double>>;
  AlignedVector center_x, center_y, center_z;
  AlignedVector radius_sq; // Store squared radius
  size_t count = 0;
};

class BatchedCollisionChecker {
public:
  BatchedCollisionChecker(const std::vector<SphereTreeNode> &object_sphere_tree,
                          const std::vector<Point3D> &obstacle_points);

  // Checks an entire path segment at once. Returns true if ANY point collides.
  bool is_path_in_collision(const std::vector<Config> &path_segment) const;

private:
  void build_soa_from_tree(const std::vector<SphereTreeNode> &tree);

  ObjectLeafSpheresSoA spheres_soa_;
  PointCloud obstacle_cloud_;
  std::unique_ptr<KDTree> obstacle_kdtree_;
};

} // namespace object_planner