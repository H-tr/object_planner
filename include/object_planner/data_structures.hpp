#pragma once

#include <Eigen/Dense>

namespace object_planner {

struct Config {
  double x, y, theta;
  Config(double x_ = 0.0, double y_ = 0.0, double t = 0.0)
      : x(x_), y(y_), theta(t) {}
};

using Point3D = Eigen::Vector3d;

struct Sphere {
  Point3D center;
  double radius;
};

struct SphereTreeNode {
  Sphere sphere;
  int parent_id = -1;
  int child1_id = -1;
  int child2_id = -1;
};

} // namespace object_planner
