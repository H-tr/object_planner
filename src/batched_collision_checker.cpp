#include "batched_collision_checker.h"
#include <Eigen/Geometry>

namespace object_planner {

BatchedCollisionChecker::BatchedCollisionChecker(
    const std::vector<SphereTreeNode> &object_sphere_tree,
    const std::vector<Point3D> &obstacle_points) {

  build_soa_from_tree(object_sphere_tree);

  obstacle_cloud_.pts = obstacle_points;
  if (!obstacle_cloud_.pts.empty()) {
    obstacle_kdtree_ = std::make_unique<KDTree>(
        3, obstacle_cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    obstacle_kdtree_->buildIndex();
  }
}

void BatchedCollisionChecker::build_soa_from_tree(
    const std::vector<SphereTreeNode> &tree) {
  for (const auto &node : tree) {
    if (node.child1_id == -1) { // It's a leaf node
      spheres_soa_.center_x.push_back(node.sphere.center.x());
      spheres_soa_.center_y.push_back(node.sphere.center.y());
      spheres_soa_.center_z.push_back(node.sphere.center.z());
      spheres_soa_.radius_sq.push_back(node.sphere.radius * node.sphere.radius);
    }
  }
  spheres_soa_.count = spheres_soa_.center_x.size();

  // Pad vectors to be a multiple of SIMD width to allow safe aligned loads
  using batch_type = xsimd::batch<double>;
  size_t remainder = spheres_soa_.count % batch_type::size;
  if (remainder != 0) {
    size_t padding = batch_type::size - remainder;
    for (size_t i = 0; i < padding; ++i) {
      spheres_soa_.center_x.push_back(0);
      spheres_soa_.center_y.push_back(0);
      spheres_soa_.center_z.push_back(0);
      spheres_soa_.radius_sq.push_back(0);
    }
  }
}

bool BatchedCollisionChecker::is_path_in_collision(
    const std::vector<Config> &path_segment) const {
  if (spheres_soa_.count == 0 || obstacle_cloud_.pts.empty()) {
    return false; // No object or no obstacles means no collision
  }

  using batch_type = xsimd::batch<double>;
  constexpr size_t simd_size = batch_type::size;

  for (const auto &config : path_segment) {
    const double s = std::sin(config.theta);
    const double c = std::cos(config.theta);
    const auto tx_b = xsimd::broadcast(config.x);
    const auto ty_b = xsimd::broadcast(config.y);
    const auto s_b = xsimd::broadcast(s);
    const auto c_b = xsimd::broadcast(c);

    // Loop over leaf spheres in SIMD-sized chunks
    for (size_t j = 0; j < spheres_soa_.center_x.size(); j += simd_size) {
      // 1. Load a batch of sphere data
      batch_type ox = xsimd::load_aligned(&spheres_soa_.center_x[j]);
      batch_type oy = xsimd::load_aligned(&spheres_soa_.center_y[j]);
      batch_type oz = xsimd::load_aligned(&spheres_soa_.center_z[j]);
      batch_type r_sq = xsimd::load_aligned(&spheres_soa_.radius_sq[j]);

      // 2. Perform 3D transformation using SIMD
      batch_type new_x = c_b * ox - s_b * oy + tx_b;
      batch_type new_y = s_b * ox + c_b * oy + ty_b;
      // Z is unchanged

      // 3. De-vectorize and check each sphere in the batch
      alignas(xsimd::default_arch::alignment()) std::array<double, simd_size>
          res_x, res_y, res_z, res_r_sq;
      xsimd::store_aligned(res_x.data(), new_x);
      xsimd::store_aligned(res_y.data(), new_y);
      xsimd::store_aligned(res_z.data(), oz);
      xsimd::store_aligned(res_r_sq.data(), r_sq);

      for (size_t k = 0; k < simd_size; ++k) {
        if (j + k >= spheres_soa_.count)
          break;

        const double search_center[3] = {res_x[k], res_y[k], res_z[k]};

        // We need a vector to hold the results, even if we ignore them.
        std::vector<nanoflann::ResultItem<unsigned int, double>> matches;

        // The function call now matches the expected signature.
        const size_t num_found =
            obstacle_kdtree_->radiusSearch(search_center, res_r_sq[k],
                                           matches, // Pass the vector here
                                           nanoflann::SearchParameters());

        if (num_found > 0) {
          return true; // Collision detected!
        }
      }
    }
  }
  return false; // No collision on entire path segment
}
} // namespace object_planner