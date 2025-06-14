#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "data_structures.h"
#include "sphere_tree_builder.h"
#include "batched_collision_checker.h"
#include "rrt_star.h"
#include "path_smoother.h"
#include <memory>
#include <cmath> // For M_PI

namespace py = pybind11;
using namespace object_planner;

// Helper to convert numpy array to vector<Point3D>
std::vector<Point3D> numpyToPoints(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    if (arr.ndim() != 2 || arr.shape(1) != 3) {
        throw std::runtime_error("Input point cloud must be an Nx3 array");
    }
    std::vector<Point3D> points(arr.shape(0));
    for (ssize_t i = 0; i < arr.shape(0); ++i) {
        points[i] = {*arr.data(i, 0), *arr.data(i, 1), *arr.data(i, 2)};
    }
    return points;
}

// Main C++ Planner class that holds all the components exposed to Python
class Planner {
public:
    Planner(const std::string& sphere_tree_file, 
            py::array_t<double> obstacle_points,
            const std::pair<double, double>& x_bounds,
            const std::pair<double, double>& y_bounds,
            const std::pair<double, double>& theta_bounds) {
        
        auto object_tree = SphereTreeBuilder::loadFromFile(sphere_tree_file);
        auto obs_pts = numpyToPoints(obstacle_points);

        checker_ = std::make_unique<BatchedCollisionChecker>(object_tree, obs_pts);
        
        Config bounds_min(x_bounds.first, y_bounds.first, theta_bounds.first);
        Config bounds_max(x_bounds.second, y_bounds.second, theta_bounds.second);
        
        rrt_planner_ = std::make_unique<RRTStarPlanner>(checker_.get(), bounds_min, bounds_max);
        smoother_ = std::make_unique<PathSmoother>(checker_.get());
    }

    std::vector<Config> plan(const Config& start, const Config& goal, 
                             const RRTStarPlanner::PlanParams& params, int smoothing_iterations) {
        auto raw_path = rrt_planner_->plan(start, goal, params);
        if (raw_path.empty()) {
            return {}; // Return empty list if no path found
        }
        return smoother_->smooth(raw_path, smoothing_iterations);
    }

    bool is_config_in_collision(const Config& config) const {
        // Use the batched checker to check a "path" of a single point.
        return checker_->is_path_in_collision({config});
    }

private:
    std::unique_ptr<BatchedCollisionChecker> checker_;
    std::unique_ptr<RRTStarPlanner> rrt_planner_;
    std::unique_ptr<PathSmoother> smoother_;
};


PYBIND11_MODULE(object_planner_py, m) {
    m.doc() = "SIMD-accelerated 3-DOF object planner with configurable bounds";

    // Bindings for data structures
    py::class_<Config>(m, "Config")
        .def(py::init<double, double, double>(), py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("theta") = 0.0)
        .def_readwrite("x", &Config::x)
        .def_readwrite("y", &Config::y)
        .def_readwrite("theta", &Config::theta)
        .def("__repr__", [](const Config &c) {
            return "<Config(x=" + std::to_string(c.x) + ", y=" + std::to_string(c.y) + ", theta=" + std::to_string(c.theta) + ")>";
        });

    py::class_<RRTStarPlanner::PlanParams>(m, "PlanParams")
        .def(py::init<>())
        .def_readwrite("max_iterations", &RRTStarPlanner::PlanParams::max_iterations)
        .def_readwrite("step_size", &RRTStarPlanner::PlanParams::step_size)
        .def_readwrite("goal_bias", &RRTStarPlanner::PlanParams::goal_bias)
        .def_readwrite("neighborhood_radius", &RRTStarPlanner::PlanParams::neighborhood_radius);

    // Bind top-level utility function
    m.def("create_sphere_tree_file", [](py::array_t<double> object_points, const std::string& filename) {
        auto points = numpyToPoints(object_points);
        auto tree = SphereTreeBuilder::build(points);
        SphereTreeBuilder::saveToFile(filename, tree);
    }, py::arg("object_points"), py::arg("filename"));

    // Bind the main Planner class with the updated constructor
    py::class_<Planner>(m, "Planner")
        .def(py::init<const std::string&, py::array_t<double>, 
                      const std::pair<double, double>&, 
                      const std::pair<double, double>&,
                      const std::pair<double, double>&>(), 
             py::arg("sphere_tree_file"), py::arg("obstacle_points"), 
             py::arg("x_bounds"), py::arg("y_bounds"),
             py::arg("theta_bounds")) // theta_bounds is now a required argument
        .def("plan", &Planner::plan, py::arg("start"), py::arg("goal"), 
             py::arg("plan_params") = RRTStarPlanner::PlanParams(), py::arg("smoothing_iterations") = 100)
        .def("is_config_in_collision", &Planner::is_config_in_collision, 
             py::arg("config"), "Checks if a single configuration is in collision.");
}