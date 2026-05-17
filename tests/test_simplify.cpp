#include <catch2/catch_test_macros.hpp>

#include "object_planner/batched_collision_checker.hpp"
#include "object_planner/simplify.hpp"
#include "object_planner/sphere_tree_builder.hpp"

#include <cmath>

using namespace object_planner;

namespace {

double path_xy_length(const std::vector<Config> &path) {
  double total = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    const double dx = path[i].x - path[i - 1].x;
    const double dy = path[i].y - path[i - 1].y;
    total += std::sqrt(dx * dx + dy * dy);
  }
  return total;
}

std::vector<SphereTreeNode> point_object() {
  return SphereTreeBuilder::build({Point3D{0.0, 0.0, 0.0}});
}

} // namespace

TEST_CASE("Shortcut shortens a deliberately wavy obstacle-free path",
          "[simplify][shortcut]") {
  auto tree = point_object();
  BatchedCollisionChecker chk(tree, {}, 0.0f);
  Simplifier simp(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});

  std::vector<Config> path{{0.0, 0.0, 0.0}, {0.25, 0.5, 0.0}, {0.5, 0.0, 0.0},
                            {0.75, 0.5, 0.0}, {1.0, 0.0, 0.0}};
  const double original = path_xy_length(path);

  SimplifySettings settings;
  settings.max_iterations = 3;
  settings.operations = {SimplifyOp::Shortcut};

  auto simplified = simp.simplify(path, settings, /*res=*/0.01);
  REQUIRE(simplified.size() >= 2);
  // Endpoints preserved.
  REQUIRE(simplified.front().x == 0.0);
  REQUIRE(simplified.back().x == 1.0);
  // Shortcut should slice the zigzag into a near-straight line.
  REQUIRE(path_xy_length(simplified) < original);
}

TEST_CASE("BSpline does not invalidate path endpoints", "[simplify][bspline]") {
  auto tree = point_object();
  BatchedCollisionChecker chk(tree, {}, 0.0f);
  Simplifier simp(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});

  std::vector<Config> path{{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  SimplifySettings settings;
  settings.max_iterations = 1;
  settings.operations = {SimplifyOp::BSpline};

  auto out = simp.simplify(path, settings, 0.01);
  REQUIRE(out.front().x == 0.0);
  REQUIRE(out.back().x == 1.0);
}

TEST_CASE("Simplify is a no-op on paths shorter than 3 waypoints",
          "[simplify]") {
  auto tree = point_object();
  BatchedCollisionChecker chk(tree, {}, 0.0f);
  Simplifier simp(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});
  SimplifySettings s;
  std::vector<Config> two{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  REQUIRE(simp.simplify(two, s).size() == 2);
}
