#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "object_planner/batched_collision_checker.hpp"
#include "object_planner/bit_star.hpp"
#include "object_planner/cost_function.hpp"
#include "object_planner/sphere_tree_builder.hpp"

#include <cmath>

using namespace object_planner;

namespace {

std::vector<SphereTreeNode> point_object() {
  return SphereTreeBuilder::build({Point3D{0.0, 0.0, 0.0}});
}

// Sum the CostFunction cost of every segment of a returned path.
double path_cost(const CostFunction &cost, const std::vector<Config> &path,
                 double resolution) {
  double total = 0.0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    total += cost.edge_cost(path[i], path[i + 1], resolution);
  }
  return total;
}

} // namespace

TEST_CASE("CostFunction stub reduces the edge cost to path length",
          "[bit_star][cost]") {
  const CostFunction cost({{0.5, 0.5, 0.0}, {1.0, 0.0, 1.0}});
  const Config a{0.0, 0.0, 0.0};
  const Config b{1.0, 2.0, 1.0};
  REQUIRE(cost.context().size() == 2);
  REQUIRE_THAT(cost.state_cost(a), Catch::Matchers::WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(cost.edge_cost(a, b, /*resolution=*/0.01),
               Catch::Matchers::WithinAbs(CostFunction::distance(a, b), 1e-9));
}

TEST_CASE("CostFunction heuristic never overestimates and the edge cost is "
          "symmetric",
          "[bit_star][cost]") {
  const CostFunction cost;
  const Config a{-0.4, 0.3, -3.0};
  const Config b{0.9, -0.7, 3.0};
  const double forward = cost.edge_cost(a, b, /*resolution=*/0.01);
  const double backward = cost.edge_cost(b, a, /*resolution=*/0.01);
  REQUIRE_THAT(forward, Catch::Matchers::WithinAbs(backward, 1e-9));
  REQUIRE(cost.heuristic_cost(a, b) <= forward + 1e-12);
  // theta wraps: these two are 0.28 rad apart, not 6.0.
  REQUIRE(CostFunction::distance({0.0, 0.0, -3.0}, {0.0, 0.0, 3.0}) < 0.1);
}

TEST_CASE("BIT* converges to the optimal straight path in an empty scene",
          "[bit_star]") {
  auto tree = point_object();
  BatchedCollisionChecker chk(tree, {}, 0.0f);
  BITStarPlanner planner(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});
  BITStarPlanner::PlanParams p;
  p.max_batches = 5;
  p.samples_per_batch = 50;

  const Config start{0.0, 0.0, 0.0};
  const Config goal{1.0, 0.0, 0.0};
  const double optimum = CostFunction::distance(start, goal);
  auto path = planner.plan(start, goal, p);
  REQUIRE(path.size() >= 2);
  REQUIRE_THAT(path.front().x, Catch::Matchers::WithinAbs(start.x, 1e-6));
  REQUIRE_THAT(path.back().x, Catch::Matchers::WithinAbs(goal.x, 1e-6));
  // Nothing is in the way, so the straight segment is the optimum. No path
  // can ever beat it: the heuristic is a hard lower bound on the true cost.
  REQUIRE(planner.solution_cost() >= optimum - 1e-9);
  // BIT* usually connects start and goal directly and lands exactly on the
  // optimum, but the goal only enters the search once it falls inside a
  // vertex's k-nearest set, so leave room for a near-optimal polyline.
  REQUIRE(planner.solution_cost() <= optimum * 1.05);
}

TEST_CASE("BIT* reports the cost of the path it returns", "[bit_star]") {
  auto tree = point_object();
  // A sparse vertical wall at x = 0.0 forces a multi-waypoint detour.
  std::vector<Point3D> obs;
  for (int i = 0; i < 7; ++i) {
    obs.emplace_back(0.0, -0.3 + i * 0.1, 0.0);
  }
  BatchedCollisionChecker chk(tree, obs, /*r=*/0.06f);
  BITStarPlanner planner(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});
  BITStarPlanner::PlanParams p;
  p.max_batches = 8;
  p.samples_per_batch = 100;

  auto path = planner.plan({-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, p);
  REQUIRE(path.size() >= 2);
  const CostFunction cost;
  REQUIRE_THAT(planner.solution_cost(),
               Catch::Matchers::WithinAbs(
                   path_cost(cost, path, p.collision_check_resolution), 1e-9));
}

TEST_CASE("BIT* respects configuration-space bounds", "[bit_star]") {
  auto tree = point_object();
  BatchedCollisionChecker chk(tree, {}, 0.0f);
  Config bmin{-1.0, -1.0, 0.0};
  Config bmax{1.0, 1.0, 2 * M_PI};
  BITStarPlanner planner(&chk, bmin, bmax);
  BITStarPlanner::PlanParams p;
  p.max_batches = 4;
  p.samples_per_batch = 50;

  auto path = planner.plan({-0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, p);
  REQUIRE(!path.empty());
  for (const auto &c : path) {
    REQUIRE(c.x >= bmin.x - 1e-6);
    REQUIRE(c.x <= bmax.x + 1e-6);
    REQUIRE(c.y >= bmin.y - 1e-6);
    REQUIRE(c.y <= bmax.y + 1e-6);
  }
}

TEST_CASE("BIT* finds a detour around a wall of obstacle points",
          "[bit_star][collision]") {
  auto tree = point_object();
  std::vector<Point3D> obs;
  for (int i = 0; i < 7; ++i) {
    obs.emplace_back(0.0, -0.3 + i * 0.1, 0.0);
  }
  BatchedCollisionChecker chk(tree, obs, /*r=*/0.06f);
  BITStarPlanner planner(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});
  BITStarPlanner::PlanParams p;
  p.max_batches = 8;
  p.samples_per_batch = 100;

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
  // The detour is longer than the blocked straight line but not absurdly so.
  REQUIRE(planner.solution_cost() > 2.0);
  REQUIRE(planner.solution_cost() < 3.0);
}

TEST_CASE("BIT* reports no path and infinite cost when the goal is walled in",
          "[bit_star][collision]") {
  auto tree = point_object();
  // Seal the goal inside a dense ring; 0.02 spacing against 0.06 inflation
  // leaves no gap for the (point) object to slip through.
  std::vector<Point3D> obs;
  for (int i = 0; i < 64; ++i) {
    const double a = 2.0 * M_PI * i / 64.0;
    obs.emplace_back(1.0 + 0.15 * std::cos(a), 0.15 * std::sin(a), 0.0);
  }
  BatchedCollisionChecker chk(tree, obs, /*r=*/0.06f);
  BITStarPlanner planner(&chk, {-2.0, -2.0, 0.0}, {2.0, 2.0, 2 * M_PI});
  BITStarPlanner::PlanParams p;
  p.max_batches = 3;
  p.samples_per_batch = 100;

  auto path = planner.plan({-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, p);
  REQUIRE(path.empty());
  REQUIRE(!std::isfinite(planner.solution_cost()));
}
