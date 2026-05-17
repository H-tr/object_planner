#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "object_planner/object_planner.hpp"

#include <memory>
#include <string>

namespace nb = nanobind;
using namespace nanobind::literals;
using namespace object_planner;

// (N, 3) C-contiguous CPU float64 array — what every NumPy caller hands us.
using NumpyPoints =
    nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

namespace {

inline std::vector<Point3D> numpy_to_points(const NumpyPoints &arr) {
  const std::size_t n = arr.shape(0);
  std::vector<Point3D> points(n);
  const double *data = arr.data();
  for (std::size_t i = 0; i < n; ++i) {
    points[i] = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};
  }
  return points;
}

// Holds the collision checker + RRT* planner + smoother together so the
// Python caller can keep a single Planner object across plan() calls.
class Planner {
public:
  Planner(const NumpyPoints &object_points, const NumpyPoints &obstacle_points,
          const std::pair<double, double> &x_bounds,
          const std::pair<double, double> &y_bounds,
          const std::pair<double, double> &theta_bounds, float point_inflation) {
    auto object_pts = numpy_to_points(object_points);
    auto object_tree = SphereTreeBuilder::build(object_pts);
    auto obs_pts = numpy_to_points(obstacle_points);
    checker_ = std::make_unique<BatchedCollisionChecker>(object_tree, obs_pts,
                                                          point_inflation);
    Config bmin(x_bounds.first, y_bounds.first, theta_bounds.first);
    Config bmax(x_bounds.second, y_bounds.second, theta_bounds.second);
    rrt_planner_ =
        std::make_unique<RRTStarPlanner>(checker_.get(), bmin, bmax);
    smoother_ = std::make_unique<PathSmoother>(checker_.get());
  }

  std::vector<Config> plan(const Config &start, const Config &goal,
                            const RRTStarPlanner::PlanParams &params,
                            int smoothing_iterations) {
    auto raw = rrt_planner_->plan(start, goal, params);
    if (raw.empty()) return {};
    return smoother_->smooth(raw, smoothing_iterations);
  }

  bool is_config_in_collision(const Config &c) const {
    return checker_->is_path_in_collision({c});
  }

private:
  std::unique_ptr<BatchedCollisionChecker> checker_;
  std::unique_ptr<RRTStarPlanner> rrt_planner_;
  std::unique_ptr<PathSmoother> smoother_;
};

} // namespace

NB_MODULE(object_planner_py, m) {
  m.doc() = "SIMD-accelerated 3-DOF object planner with configurable bounds";

  nb::class_<Config>(m, "Config")
      .def(nb::init<double, double, double>(), "x"_a = 0.0, "y"_a = 0.0,
           "theta"_a = 0.0)
      .def_rw("x", &Config::x)
      .def_rw("y", &Config::y)
      .def_rw("theta", &Config::theta)
      .def("__repr__", [](const Config &c) {
        return "<Config(x=" + std::to_string(c.x) + ", y=" + std::to_string(c.y) +
               ", theta=" + std::to_string(c.theta) + ")>";
      });

  nb::class_<RRTStarPlanner::PlanParams>(m, "PlanParams")
      .def(nb::init<>())
      .def_rw("max_iterations", &RRTStarPlanner::PlanParams::max_iterations)
      .def_rw("step_size", &RRTStarPlanner::PlanParams::step_size)
      .def_rw("goal_bias", &RRTStarPlanner::PlanParams::goal_bias)
      .def_rw("neighborhood_radius",
              &RRTStarPlanner::PlanParams::neighborhood_radius);

  nb::class_<Planner>(m, "Planner")
      .def(nb::init<const NumpyPoints &, const NumpyPoints &,
                    const std::pair<double, double> &,
                    const std::pair<double, double> &,
                    const std::pair<double, double> &, float>(),
           "object_points"_a, "obstacle_points"_a, "x_bounds"_a,
           "y_bounds"_a, "theta_bounds"_a, "point_inflation"_a = 0.0f)
      .def("plan", &Planner::plan, "start"_a, "goal"_a,
           "plan_params"_a = RRTStarPlanner::PlanParams(),
           "smoothing_iterations"_a = 100)
      .def("is_config_in_collision", &Planner::is_config_in_collision,
           "config"_a, "Checks if a single configuration is in collision.");
}
