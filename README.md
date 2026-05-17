# 3D Object Planner

A **header-only**, SIMD-accelerated 3-DOF (x, y, theta) motion planner for
moving a single rigid object through a tabletop scene of obstacle point
clouds. Drop the headers into any C++ project, or use the bundled
nanobind Python module.

## Features

- **Header-only C++17**: `#include <object_planner/object_planner.hpp>` —
  no built library required for downstream consumers.
- **Sphere-vs-sphere collision with per-cloud inflation**: every
  obstacle point is treated as a sphere of configurable radius. The
  object is decomposed into a BVH of bounding spheres; only the leaves
  are consulted at query time.
- **Genuinely SIMD inner loop**: obstacle points live in a
  structure-of-arrays BSP tree; the per-leaf distance check is a single
  `xsimd::batch<float>` reduction (8-wide on AVX2, 4-wide on NEON).
- **RRT\***: optimal sampling-based motion planning. (Will be replaced
  with RRT-Connect + VAMP-style simplification in upcoming milestones.)
- **nanobind Python module**: lightweight bindings, no pybind11
  dependency.

## C++ integration (header-only)

```cmake
add_subdirectory(third_party/object_planner)
target_link_libraries(my_target PRIVATE object_planner::object_planner)
```

```cpp
#include <object_planner/object_planner.hpp>
using namespace object_planner;

auto tree = SphereTreeBuilder::build(object_points);
BatchedCollisionChecker checker(tree, obstacle_points, /*point_inflation=*/0.012f);
RRTStarPlanner planner(&checker, bounds_min, bounds_max);
auto path = planner.plan(start, goal, RRTStarPlanner::PlanParams{});
```

Required: Eigen 3.3+ and a C++17 compiler. `xsimd` is fetched
automatically via `FetchContent` when you `add_subdirectory()`.

## Python install

```bash
pip install -e third_party/object_planner   # editable build
# or, inside this repo's pixi env:
pixi install
```

## Run the demo

```bash
python third_party/object_planner/examples/run_planner.py
```
