#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "object_planner/object_planner.hpp"

#include <memory>
#include <optional>
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

// Holds the collision checker + RRT-Connect planner + BIT* planner +
// VAMP-style simplifier together. Python callers keep a single Planner
// across plan() calls so the BVH and BSP tree are only built once.
class Planner {
public:
  Planner(const NumpyPoints &object_points, const NumpyPoints &obstacle_points,
          const std::pair<double, double> &x_bounds,
          const std::pair<double, double> &y_bounds,
          const std::pair<double, double> &theta_bounds, float point_inflation,
          std::vector<Config> cost_context) {
    auto object_pts = numpy_to_points(object_points);
    auto object_tree = SphereTreeBuilder::build(object_pts);
    auto obs_pts = numpy_to_points(obstacle_points);
    checker_ = std::make_unique<BatchedCollisionChecker>(object_tree, obs_pts,
                                                          point_inflation);
    Config bmin(x_bounds.first, y_bounds.first, theta_bounds.first);
    Config bmax(x_bounds.second, y_bounds.second, theta_bounds.second);
    rrt_planner_ =
        std::make_unique<RRTConnectPlanner>(checker_.get(), bmin, bmax);
    bit_star_planner_ = std::make_unique<BITStarPlanner>(
        checker_.get(), bmin, bmax, std::move(cost_context));
    simplifier_ = std::make_unique<Simplifier>(checker_.get(), bmin, bmax);
  }

  std::vector<Config> plan(const Config &start, const Config &goal,
                            const RRTConnectPlanner::PlanParams &params,
                            std::optional<SimplifySettings> simplify_settings) {
    auto raw = rrt_planner_->plan(start, goal, params);
    if (raw.empty()) return {};
    if (simplify_settings.has_value()) {
      return simplifier_->simplify(std::move(raw), simplify_settings.value(),
                                    params.collision_check_resolution);
    }
    return raw;
  }

  std::vector<Config> plan_bit_star(const Config &start, const Config &goal,
                                      const BITStarPlanner::PlanParams &params) {
    return bit_star_planner_->plan(start, goal, params);
  }

  double bit_star_solution_cost() const {
    return bit_star_planner_->solution_cost();
  }

  bool is_config_in_collision(const Config &c) const {
    return checker_->is_path_in_collision({c});
  }

private:
  std::unique_ptr<BatchedCollisionChecker> checker_;
  std::unique_ptr<RRTConnectPlanner> rrt_planner_;
  std::unique_ptr<BITStarPlanner> bit_star_planner_;
  std::unique_ptr<Simplifier> simplifier_;
};

} // namespace

NB_MODULE(object_planner_py, m) {
  m.doc() = "SIMD-accelerated 3-DOF object planner (RRT-Connect + BIT* + VAMP "
            "simplify)";

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

  nb::class_<RRTConnectPlanner::PlanParams>(m, "PlanParams")
      .def(nb::init<>())
      .def_rw("max_iterations", &RRTConnectPlanner::PlanParams::max_iterations)
      .def_rw("step_size", &RRTConnectPlanner::PlanParams::step_size)
      .def_rw("collision_check_resolution",
              &RRTConnectPlanner::PlanParams::collision_check_resolution);

  nb::class_<BITStarPlanner::PlanParams>(m, "BITStarParams")
      .def(nb::init<>())
      .def_rw("max_batches", &BITStarPlanner::PlanParams::max_batches)
      .def_rw("samples_per_batch",
              &BITStarPlanner::PlanParams::samples_per_batch)
      .def_rw("collision_check_resolution",
              &BITStarPlanner::PlanParams::collision_check_resolution);

  nb::enum_<SimplifyOp>(m, "SimplifyOp")
      .value("Shortcut", SimplifyOp::Shortcut)
      .value("Reduce", SimplifyOp::Reduce)
      .value("Perturb", SimplifyOp::Perturb)
      .value("BSpline", SimplifyOp::BSpline);

  nb::class_<ShortcutSettings>(m, "ShortcutSettings").def(nb::init<>());

  nb::class_<ReduceSettings>(m, "ReduceSettings")
      .def(nb::init<>())
      .def_rw("max_steps", &ReduceSettings::max_steps)
      .def_rw("max_empty_steps", &ReduceSettings::max_empty_steps)
      .def_rw("range_ratio", &ReduceSettings::range_ratio);

  nb::class_<PerturbSettings>(m, "PerturbSettings")
      .def(nb::init<>())
      .def_rw("max_steps", &PerturbSettings::max_steps)
      .def_rw("max_empty_steps", &PerturbSettings::max_empty_steps)
      .def_rw("perturbation_attempts", &PerturbSettings::perturbation_attempts)
      .def_rw("range", &PerturbSettings::range);

  nb::class_<BSplineSettings>(m, "BSplineSettings")
      .def(nb::init<>())
      .def_rw("max_steps", &BSplineSettings::max_steps)
      .def_rw("min_change", &BSplineSettings::min_change)
      .def_rw("midpoint_interpolation",
              &BSplineSettings::midpoint_interpolation);

  nb::class_<SimplifySettings>(m, "SimplifySettings")
      .def(nb::init<>())
      .def_rw("max_iterations", &SimplifySettings::max_iterations)
      .def_rw("operations", &SimplifySettings::operations)
      .def_rw("shortcut", &SimplifySettings::shortcut)
      .def_rw("reduce", &SimplifySettings::reduce)
      .def_rw("perturb", &SimplifySettings::perturb)
      .def_rw("bspline", &SimplifySettings::bspline);

  nb::class_<Planner>(m, "Planner")
      .def(nb::init<const NumpyPoints &, const NumpyPoints &,
                    const std::pair<double, double> &,
                    const std::pair<double, double> &,
                    const std::pair<double, double> &, float,
                    std::vector<Config>>(),
           "object_points"_a, "obstacle_points"_a, "x_bounds"_a,
           "y_bounds"_a, "theta_bounds"_a, "point_inflation"_a = 0.0f,
           "cost_context"_a = std::vector<Config>(),
           "cost_context is the list of (x, y, theta) dependencies BIT*'s "
           "cost function reads; it is fixed here, at setup.")
      .def("plan", &Planner::plan, "start"_a, "goal"_a,
           "plan_params"_a = RRTConnectPlanner::PlanParams(),
           "simplify_settings"_a = nb::none(),
           "Plan a path. If simplify_settings is None, the raw RRT-Connect "
           "path is returned; otherwise it is post-processed by the "
           "VAMP-style simplifier.")
      .def("plan_bit_star", &Planner::plan_bit_star, "start"_a, "goal"_a,
           "plan_params"_a = BITStarPlanner::PlanParams(),
           "Plan the cheapest path under the cost function with BIT*. The "
           "result is deliberately not simplified: shortcutting optimises "
           "length, which would undo the cost function's shaping.")
      .def("bit_star_solution_cost", &Planner::bit_star_solution_cost,
           "Cost of the path plan_bit_star() last returned; infinity if none.")
      .def("is_config_in_collision", &Planner::is_config_in_collision,
           "config"_a, "Checks if a single configuration is in collision.");
}
