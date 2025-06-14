#pragma once

#include "data_structures.h"
#include <string>
#include <vector>

namespace object_planner {

class SphereTreeBuilder {
public:
  static std::vector<SphereTreeNode> build(const std::vector<Point3D> &points);
  static void saveToFile(const std::string &filename,
                         const std::vector<SphereTreeNode> &tree);
  static std::vector<SphereTreeNode> loadFromFile(const std::string &filename);

private:
  static int buildRecursive(std::vector<Point3D> &points,
                            std::vector<SphereTreeNode> &tree, int parent_id);
  static Sphere computeBoundingSphere(const std::vector<Point3D> &points);
};

} // namespace object_planner