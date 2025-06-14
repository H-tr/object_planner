#include "sphere_tree_builder.h"
#include <fstream>
#include <iostream>
#include <numeric>

namespace object_planner {

// Helper function to get the principal axis for splitting points
Eigen::Vector3d getPrincipalAxis(const std::vector<Point3D> &points) {
  if (points.size() <= 1) {
    return Eigen::Vector3d::UnitX();
  }
  Eigen::MatrixXd centered = Eigen::MatrixXd::Zero(3, points.size());
  Point3D mean =
      std::accumulate(points.begin(), points.end(), Point3D::Zero().eval()) /
      points.size();
  for (size_t i = 0; i < points.size(); ++i) {
    centered.col(i) = points[i] - mean;
  }
  Eigen::Matrix3d cov = centered * centered.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);
  return eigensolver.eigenvectors().col(
      2); // Vector corresponding to largest eigenvalue
}

Sphere
SphereTreeBuilder::computeBoundingSphere(const std::vector<Point3D> &points) {
  if (points.empty())
    return {{0, 0, 0}, 0};

  Point3D center =
      std::accumulate(points.begin(), points.end(), Point3D::Zero().eval()) /
      points.size();

  double max_dist_sq = 0.0;
  for (const auto &p : points) {
    max_dist_sq = std::max(max_dist_sq, (p - center).squaredNorm());
  }
  return {center, std::sqrt(max_dist_sq)};
}

int SphereTreeBuilder::buildRecursive(std::vector<Point3D> &points,
                                      std::vector<SphereTreeNode> &tree,
                                      int parent_id) {
  if (points.empty())
    return -1;

  int current_node_id = tree.size();
  tree.emplace_back();
  tree[current_node_id].parent_id = parent_id;
  tree[current_node_id].sphere = computeBoundingSphere(points);

  if (points.size() < 10) { // Leaf node condition
    return current_node_id;
  }

  Point3D center = tree[current_node_id].sphere.center;
  Eigen::Vector3d axis = getPrincipalAxis(points);

  std::vector<Point3D> left_points, right_points;
  for (const auto &p : points) {
    if ((p - center).dot(axis) <= 0) {
      left_points.push_back(p);
    } else {
      right_points.push_back(p);
    }
  }

  // Avoid pathological splits
  if (left_points.empty() || right_points.empty()) {
    return current_node_id;
  }

  tree[current_node_id].child1_id =
      buildRecursive(left_points, tree, current_node_id);
  tree[current_node_id].child2_id =
      buildRecursive(right_points, tree, current_node_id);

  return current_node_id;
}

std::vector<SphereTreeNode>
SphereTreeBuilder::build(const std::vector<Point3D> &points) {
  std::vector<SphereTreeNode> tree;
  if (points.empty())
    return tree;
  auto points_copy = points;
  buildRecursive(points_copy, tree, -1);
  return tree;
}

void SphereTreeBuilder::saveToFile(const std::string &filename,
                                   const std::vector<SphereTreeNode> &tree) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  size_t tree_size = tree.size();
  out.write(reinterpret_cast<const char *>(&tree_size), sizeof(tree_size));
  out.write(reinterpret_cast<const char *>(tree.data()),
            tree_size * sizeof(SphereTreeNode));
}

std::vector<SphereTreeNode>
SphereTreeBuilder::loadFromFile(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }
  size_t tree_size;
  in.read(reinterpret_cast<char *>(&tree_size), sizeof(tree_size));
  std::vector<SphereTreeNode> tree(tree_size);
  in.read(reinterpret_cast<char *>(tree.data()),
          tree_size * sizeof(SphereTreeNode));
  return tree;
}

} // namespace object_planner