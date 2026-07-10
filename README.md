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
- **RRT-Connect**: fast bidirectional planning for a first feasible path,
  plus a VAMP-style `Simplifier` to shorten it.
- **BIT\***: anytime, asymptotically optimal planning that minimises a
  user-supplied cost function.
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

// First feasible path, then shortened.
RRTConnectPlanner planner(&checker, bounds_min, bounds_max);
auto path = planner.plan(start, goal, RRTConnectPlanner::PlanParams{});
Simplifier simplifier(&checker, bounds_min, bounds_max);
path = simplifier.simplify(path, SimplifySettings{});
```

Required: Eigen 3.3+ and a C++17 compiler. `xsimd` is fetched
automatically via `FetchContent` when you `add_subdirectory()`.

## Planning against a cost function (BIT\*)

`BITStarPlanner` returns the cheapest path it can find rather than the
first feasible one. It minimises the line integral of `1 + state_cost`
along the path, so an edge costs the distance it covers plus whatever
extra `CostFunction::state_cost` charges for the configurations it
passes through.

```cpp
// The (x, y, theta) dependencies the cost needs, fixed at setup.
std::vector<Config> cost_context = {{0.4, 0.1, 0.0}, {0.9, -0.2, 1.57}};
BITStarPlanner planner(&checker, bounds_min, bounds_max, cost_context);
auto path = planner.plan(start, goal, BITStarPlanner::PlanParams{});
const double cost = planner.solution_cost();  // infinity if no path
```

`CostFunction::state_cost` in `include/object_planner/cost_function.hpp`
is a stub returning `0.0`, which makes BIT\* minimise plain path length.
It is the only function to implement; it must stay finite and
non-negative, which is what keeps `CostFunction::heuristic_cost` an
admissible lower bound and BIT\* convergent.

From Python, the context is a list of `Config`:

```python
planner = opp.Planner(..., cost_context=[opp.Config(0.4, 0.1, 0.0)])
path = planner.plan_bit_star(start, goal, opp.BITStarParams())
cost = planner.bit_star_solution_cost()
```

## C++ tests

```bash
cmake -S . -B build -DOBJECT_PLANNER_BUILD_TESTS=ON -DOBJECT_PLANNER_BUILD_PYTHON=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Python install

Build into a virtual environment; the extension is compiled on install, so
you need a C++17 compiler and Eigen 3.3+ on the system.

```bash
python3 -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -e .                                     # editable build
```

## Run the demo

The demos render with Open3D, which is not a library dependency:

```bash
pip install open3d
python examples/run_rrtc_planner.py   # RRT-Connect + simplify
python examples/run_bit_star.py       # BIT*, sweeping the batch budget
```

`run_bit_star.py --no-viz` plans and prints without opening a window.
