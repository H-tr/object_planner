#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "object_planner/batched_collision_checker.hpp"
#include "object_planner/sphere_tree_builder.hpp"

#include <vector>

using namespace object_planner;

namespace {

// A two-point object: leaf-sphere radius ends up being the half-distance
// between the two points. With far-apart obstacle points this gives us a
// non-trivial sphere-vs-sphere check.
std::vector<SphereTreeNode> single_point_object(double x, double y, double z) {
  std::vector<Point3D> pts{{x, y, z}};
  return SphereTreeBuilder::build(pts);
}

} // namespace

TEST_CASE("Empty obstacles yields no collision", "[checker]") {
  auto tree = single_point_object(0.0, 0.0, 0.0);
  BatchedCollisionChecker chk(tree, {}, /*point_inflation=*/0.1f);
  REQUIRE(!chk.is_path_in_collision({Config{0.0, 0.0, 0.0}}));
}

TEST_CASE("Empty object yields no collision", "[checker]") {
  std::vector<Point3D> obs{{0.0, 0.0, 0.0}};
  BatchedCollisionChecker chk({}, obs, 0.1f);
  REQUIRE(!chk.is_path_in_collision({Config{0.0, 0.0, 0.0}}));
}

TEST_CASE("Inflation gates whether a near point counts as collision",
          "[checker][inflation]") {
  auto tree = single_point_object(0.0, 0.0, 0.0);
  std::vector<Point3D> obs{{0.05, 0.0, 0.0}};  // 5 cm away

  SECTION("inflation < gap → no collision") {
    BatchedCollisionChecker chk(tree, obs, /*r=*/0.04f);
    REQUIRE(!chk.is_path_in_collision({Config{0.0, 0.0, 0.0}}));
  }
  SECTION("inflation > gap → collision") {
    BatchedCollisionChecker chk(tree, obs, /*r=*/0.06f);
    REQUIRE(chk.is_path_in_collision({Config{0.0, 0.0, 0.0}}));
  }
}

TEST_CASE("SE(2) translation moves the object away from the obstacle",
          "[checker][se2]") {
  auto tree = single_point_object(0.0, 0.0, 0.0);
  std::vector<Point3D> obs{{0.0, 0.0, 0.0}};
  BatchedCollisionChecker chk(tree, obs, 0.05f);
  REQUIRE(chk.is_path_in_collision({Config{0.0, 0.0, 0.0}}));    // overlap
  REQUIRE(!chk.is_path_in_collision({Config{1.0, 0.0, 0.0}}));   // 1 m away
}

TEST_CASE("SE(2) rotation transforms object points before the test",
          "[checker][se2][rotation]") {
  // Object: single point at (0.1, 0, 0) in object frame.
  // Obstacle: single point at world (0, 0.1, 0).
  auto tree = single_point_object(0.1, 0.0, 0.0);
  std::vector<Point3D> obs{{0.0, 0.1, 0.0}};
  BatchedCollisionChecker chk(tree, obs, /*r=*/0.01f);

  // Identity config: object point at (0.1, 0, 0). Obstacle at (0, 0.1, 0).
  // Gap = sqrt(0.02) ≈ 0.141 > 0.01 inflation → no collision.
  REQUIRE(!chk.is_path_in_collision({Config{0.0, 0.0, 0.0}}));
  // θ = π/2: object point rotates to (0, 0.1, 0) — overlaps the obstacle.
  REQUIRE(chk.is_path_in_collision({Config{0.0, 0.0, M_PI / 2}}));
}

TEST_CASE("Path-segment query returns true if any config collides",
          "[checker][path]") {
  auto tree = single_point_object(0.0, 0.0, 0.0);
  std::vector<Point3D> obs{{0.5, 0.0, 0.0}};
  BatchedCollisionChecker chk(tree, obs, 0.02f);
  // 3-config path; middle config sits on the obstacle.
  std::vector<Config> path{
      {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  REQUIRE(chk.is_path_in_collision(path));
}

TEST_CASE("BSP traversal still finds collisions in a dense obstacle field",
          "[checker][bsp]") {
  // Many obstacle points so the BSP definitely splits into multiple leaves.
  std::vector<Point3D> obs;
  for (int i = 0; i < 200; ++i) {
    obs.emplace_back(0.01 * i, 0.0, 0.0);  // 2-metre line along +x
  }
  auto tree = single_point_object(0.0, 0.0, 0.0);
  BatchedCollisionChecker chk(tree, obs, 0.005f);
  // Sample a config whose object point sits exactly on an obstacle point.
  REQUIRE(chk.is_path_in_collision({Config{1.23, 0.0, 0.0}}));
  // A config off to the side stays clear.
  REQUIRE(!chk.is_path_in_collision({Config{1.0, 1.0, 0.0}}));
}
