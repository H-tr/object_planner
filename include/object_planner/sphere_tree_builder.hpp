#pragma once

#include "data_structures.hpp"

#include <Eigen/Eigenvalues>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace object_planner {

// Top-down BVH builder. Each node holds a bounding sphere; leaves (the
// only spheres consulted by the collision checker) contain at most
// SphereTreeBuilder::leaf_size_threshold points each. Splits along the
// principal eigenvector of the local point-cloud covariance.
class SphereTreeBuilder {
public:
  static constexpr std::size_t leaf_size_threshold = 10;

  static inline std::vector<SphereTreeNode>
  build(const std::vector<Point3D> &points);

  // Optional disk roundtrip. Kept so downstream C++ consumers can cache
  // an expensive build; the Python binding stops using it in milestone 2.
  static inline void saveToFile(const std::string &filename,
                                const std::vector<SphereTreeNode> &tree);
  static inline std::vector<SphereTreeNode>
  loadFromFile(const std::string &filename);

private:
  static inline int buildRecursive(std::vector<Point3D> &points,
                                    std::vector<SphereTreeNode> &tree,
                                    int parent_id);
  static inline Sphere
  computeBoundingSphere(const std::vector<Point3D> &points);
  static inline Eigen::Vector3d
  principalAxis(const std::vector<Point3D> &points);
};

inline Eigen::Vector3d
SphereTreeBuilder::principalAxis(const std::vector<Point3D> &points) {
  if (points.size() <= 1) {
    return Eigen::Vector3d::UnitX();
  }
  Eigen::MatrixXd centered = Eigen::MatrixXd::Zero(3, points.size());
  Point3D mean =
      std::accumulate(points.begin(), points.end(), Point3D::Zero().eval()) /
      static_cast<double>(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    centered.col(i) = points[i] - mean;
  }
  Eigen::Matrix3d cov = centered * centered.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);
  return eigensolver.eigenvectors().col(2);
}

inline Sphere
SphereTreeBuilder::computeBoundingSphere(const std::vector<Point3D> &points) {
  if (points.empty()) {
    return {{0, 0, 0}, 0};
  }
  Point3D center =
      std::accumulate(points.begin(), points.end(), Point3D::Zero().eval()) /
      static_cast<double>(points.size());
  double max_dist_sq = 0.0;
  for (const auto &p : points) {
    max_dist_sq = std::max(max_dist_sq, (p - center).squaredNorm());
  }
  return {center, std::sqrt(max_dist_sq)};
}

inline int SphereTreeBuilder::buildRecursive(std::vector<Point3D> &points,
                                              std::vector<SphereTreeNode> &tree,
                                              int parent_id) {
  if (points.empty()) {
    return -1;
  }
  const int current_node_id = static_cast<int>(tree.size());
  tree.emplace_back();
  tree[current_node_id].parent_id = parent_id;
  tree[current_node_id].sphere = computeBoundingSphere(points);

  if (points.size() < leaf_size_threshold) {
    return current_node_id;
  }

  const Point3D center = tree[current_node_id].sphere.center;
  const Eigen::Vector3d axis = principalAxis(points);
  std::vector<Point3D> left_points, right_points;
  for (const auto &p : points) {
    if ((p - center).dot(axis) <= 0) {
      left_points.push_back(p);
    } else {
      right_points.push_back(p);
    }
  }
  if (left_points.empty() || right_points.empty()) {
    return current_node_id;
  }
  tree[current_node_id].child1_id =
      buildRecursive(left_points, tree, current_node_id);
  tree[current_node_id].child2_id =
      buildRecursive(right_points, tree, current_node_id);
  return current_node_id;
}

inline std::vector<SphereTreeNode>
SphereTreeBuilder::build(const std::vector<Point3D> &points) {
  std::vector<SphereTreeNode> tree;
  if (points.empty()) {
    return tree;
  }
  auto points_copy = points;
  buildRecursive(points_copy, tree, -1);
  return tree;
}

inline void
SphereTreeBuilder::saveToFile(const std::string &filename,
                              const std::vector<SphereTreeNode> &tree) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  std::size_t tree_size = tree.size();
  out.write(reinterpret_cast<const char *>(&tree_size), sizeof(tree_size));
  out.write(reinterpret_cast<const char *>(tree.data()),
            tree_size * sizeof(SphereTreeNode));
}

inline std::vector<SphereTreeNode>
SphereTreeBuilder::loadFromFile(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }
  std::size_t tree_size;
  in.read(reinterpret_cast<char *>(&tree_size), sizeof(tree_size));
  std::vector<SphereTreeNode> tree(tree_size);
  in.read(reinterpret_cast<char *>(tree.data()),
          tree_size * sizeof(SphereTreeNode));
  return tree;
}

} // namespace object_planner
