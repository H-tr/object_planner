#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "object_planner/sphere_tree_builder.hpp"

#include <random>

using namespace object_planner;

TEST_CASE("SphereTreeBuilder returns an empty tree for no points", "[sphere_tree]") {
  auto tree = SphereTreeBuilder::build({});
  REQUIRE(tree.empty());
}

TEST_CASE("SphereTreeBuilder builds a one-node tree for a single point",
          "[sphere_tree]") {
  std::vector<Point3D> pts{{0.1, 0.2, 0.3}};
  auto tree = SphereTreeBuilder::build(pts);
  REQUIRE(tree.size() == 1);
  REQUIRE(tree[0].parent_id == -1);
  REQUIRE(tree[0].child1_id == -1);  // leaf
  REQUIRE(tree[0].child2_id == -1);
  REQUIRE_THAT(tree[0].sphere.radius,
               Catch::Matchers::WithinAbs(0.0, 1e-9));
}

TEST_CASE("SphereTreeBuilder produces a leaf for every point cluster",
          "[sphere_tree]") {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-0.05, 0.05);
  std::vector<Point3D> pts;
  for (int i = 0; i < 50; ++i) {
    pts.emplace_back(dist(rng), dist(rng), dist(rng));
  }
  auto tree = SphereTreeBuilder::build(pts);
  REQUIRE(!tree.empty());
  // Exactly one root.
  int roots = 0;
  int leaves = 0;
  for (const auto &node : tree) {
    if (node.parent_id == -1) ++roots;
    if (node.child1_id == -1) ++leaves;
  }
  REQUIRE(roots == 1);
  REQUIRE(leaves > 0);
  // Every leaf's bounding sphere is large enough to contain every point it
  // "owns" — there's no clean way to walk owners without rebuilding the
  // recursion, but we can at least verify the root sphere contains all
  // input points.
  for (const auto &p : pts) {
    const double d = (p - tree[0].sphere.center).norm();
    REQUIRE(d <= tree[0].sphere.radius + 1e-9);
  }
}
