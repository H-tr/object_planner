#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "object_planner/batched_collision_checker.hpp"
#include "object_planner/rrt_connect.hpp"
#include "object_planner/sphere_tree_builder.hpp"

#include <cmath>

using namespace object_planner;

namespace {

std::vector<SphereTreeNode> point_object() {
  return SphereTreeBuilder::build({Point3D{0.0, 0.0, 0.0}});
}

} // namespace

TEST_CASE("RRT-Connect plan from start to goal in empty scene",
          "[rrt_connect]") {
  auto tree = point_object();
  BatchedCollisionChecker chk(tree, {}, 0.0f);
  RRTConnectPlanner planner(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});
  RRTConnectPlanner::PlanParams p;
  p.max_iterations = 2000;
  p.step_size = 0.1;
  auto path = planner.plan({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, p);
  REQUIRE(path.size() >= 2);
  // Endpoints match start and goal.
  REQUIRE_THAT(path.front().x, Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(path.back().x, Catch::Matchers::WithinAbs(1.0, 1e-6));
}

TEST_CASE("RRT-Connect respects configuration-space bounds", "[rrt_connect]") {
  auto tree = point_object();
  BatchedCollisionChecker chk(tree, {}, 0.0f);
  Config bmin{-1.0, -1.0, 0.0};
  Config bmax{1.0, 1.0, 2 * M_PI};
  RRTConnectPlanner planner(&chk, bmin, bmax);
  RRTConnectPlanner::PlanParams p;
  p.max_iterations = 1000;
  p.step_size = 0.05;
  auto path = planner.plan({-0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, p);
  REQUIRE(!path.empty());
  for (const auto &c : path) {
    REQUIRE(c.x >= bmin.x - 1e-6);
    REQUIRE(c.x <= bmax.x + 1e-6);
    REQUIRE(c.y >= bmin.y - 1e-6);
    REQUIRE(c.y <= bmax.y + 1e-6);
  }
}

TEST_CASE("RRT-Connect finds a detour around a wall of obstacle points",
          "[rrt_connect][collision]") {
  auto tree = point_object();
  // A sparse vertical wall at x = 0.0 covering y ∈ [-0.3, 0.3].
  // With 5 cm inflation each point gates the corridor.
  std::vector<Point3D> obs;
  for (int i = 0; i < 7; ++i) {
    obs.emplace_back(0.0, -0.3 + i * 0.1, 0.0);
  }
  BatchedCollisionChecker chk(tree, obs, /*r=*/0.06f);
  RRTConnectPlanner planner(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});
  RRTConnectPlanner::PlanParams p;
  p.max_iterations = 8000;
  p.step_size = 0.05;
  auto path = planner.plan({-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, p);
  REQUIRE(path.size() >= 2);
  // Every waypoint must clear every wall point by at least the inflation.
  for (const auto &c : path) {
    for (const auto &o : obs) {
      const double dx = c.x - o.x();
      const double dy = c.y - o.y();
      REQUIRE(std::sqrt(dx * dx + dy * dy) >= 0.05);
    }
  }
}
