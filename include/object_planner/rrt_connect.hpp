#pragma once

#include "batched_collision_checker.hpp"
#include "data_structures.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace object_planner {

struct RRTConnectNode {
  Config config;
  int parent_id = -1;
};

// Bidirectional RRT-Connect: grow two trees (one from start, one from
// goal) toward each other. Each iteration extends one tree by a small
// step toward a random sample, then the *other* tree tries to connect
// to that new configuration with as many steps as the local clearance
// allows. Returns the first feasible path found; empty if the iteration
// budget is exhausted.
class RRTConnectPlanner {
public:
  struct PlanParams {
    int max_iterations = 8000;
    double step_size = 0.05;
    double collision_check_resolution = 0.01;
  };

  enum class ExtendResult { Trapped, Advanced, Reached };

  inline RRTConnectPlanner(const BatchedCollisionChecker *checker,
                            Config bounds_min, Config bounds_max);

  inline std::vector<Config> plan(const Config &start, const Config &goal,
                                    const PlanParams &params);

private:
  struct Tree {
    std::vector<RRTConnectNode> nodes;
  };

  inline Config sample_random_config();
  inline int find_nearest(const Tree &tree, const Config &config);
  inline Config steer(const Config &from, const Config &to, double step_size);
  inline bool is_path_collision_free(const Config &from, const Config &to,
                                       double resolution);
  inline ExtendResult extend(Tree &tree, const Config &q, double step_size,
                              double resolution);
  inline ExtendResult connect(Tree &tree, const Config &q, double step_size,
                               double resolution);
  inline double calculate_distance(const Config &from, const Config &to);
  inline double normalize_angle(double angle);
  inline std::vector<Config> reconstruct(const Tree &tree, int node_id);

  const BatchedCollisionChecker *checker_;
  Config bounds_min_, bounds_max_;
  std::mt19937 random_engine_;
  std::uniform_real_distribution<double> uniform_dist_;
};

inline RRTConnectPlanner::RRTConnectPlanner(
    const BatchedCollisionChecker *checker, Config bounds_min, Config bounds_max)
    : checker_(checker), bounds_min_(bounds_min), bounds_max_(bounds_max),
      random_engine_(std::random_device{}()), uniform_dist_(0.0, 1.0) {}

inline std::vector<Config>
RRTConnectPlanner::plan(const Config &start, const Config &goal,
                          const PlanParams &params) {
  Tree tree_a, tree_b;
  tree_a.nodes.push_back({start, -1});
  tree_b.nodes.push_back({goal, -1});
  bool a_is_start_side = true;

  for (int iter = 0; iter < params.max_iterations; ++iter) {
    const Config q_rand = sample_random_config();
    const ExtendResult er =
        extend(tree_a, q_rand, params.step_size,
               params.collision_check_resolution);
    if (er != ExtendResult::Trapped) {
      const Config q_new = tree_a.nodes.back().config;
      if (connect(tree_b, q_new, params.step_size,
                  params.collision_check_resolution) == ExtendResult::Reached) {
        const int a_tail = static_cast<int>(tree_a.nodes.size() - 1);
        const int b_tail = static_cast<int>(tree_b.nodes.size() - 1);
        std::vector<Config> path_a = reconstruct(tree_a, a_tail);
        std::vector<Config> path_b = reconstruct(tree_b, b_tail);
        // path_a goes q_new -> root_of_a; path_b goes q_new -> root_of_b.
        // Combine into root_of_a -> q_new -> root_of_b. Drop one copy of
        // q_new at the join.
        std::reverse(path_a.begin(), path_a.end());
        if (!path_b.empty()) path_b.erase(path_b.begin());
        path_a.insert(path_a.end(), path_b.begin(), path_b.end());
        if (!a_is_start_side) {
          std::reverse(path_a.begin(), path_a.end());
        }
        return path_a;
      }
    }
    std::swap(tree_a, tree_b);
    a_is_start_side = !a_is_start_side;
  }
  return {};
}

inline Config RRTConnectPlanner::sample_random_config() {
  return {uniform_dist_(random_engine_) * (bounds_max_.x - bounds_min_.x) +
              bounds_min_.x,
          uniform_dist_(random_engine_) * (bounds_max_.y - bounds_min_.y) +
              bounds_min_.y,
          uniform_dist_(random_engine_) *
                  (bounds_max_.theta - bounds_min_.theta) +
              bounds_min_.theta};
}

inline int RRTConnectPlanner::find_nearest(const Tree &tree,
                                             const Config &config) {
  double min_dist = std::numeric_limits<double>::infinity();
  int nearest_id = -1;
  for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
    const double d = calculate_distance(tree.nodes[i].config, config);
    if (d < min_dist) {
      min_dist = d;
      nearest_id = static_cast<int>(i);
    }
  }
  return nearest_id;
}

inline Config RRTConnectPlanner::steer(const Config &from, const Config &to,
                                         double step_size) {
  const double dist = calculate_distance(from, to);
  if (dist <= step_size) {
    return to;
  }
  const double ratio = step_size / dist;
  const double delta_theta = normalize_angle(to.theta - from.theta);
  return {from.x + ratio * (to.x - from.x), from.y + ratio * (to.y - from.y),
          normalize_angle(from.theta + ratio * delta_theta)};
}

inline bool RRTConnectPlanner::is_path_collision_free(const Config &from,
                                                        const Config &to,
                                                        double resolution) {
  std::vector<Config> path_segment;
  const double dist = calculate_distance(from, to);
  int num_steps = static_cast<int>(dist / resolution);
  if (num_steps < 2) num_steps = 2;
  for (int i = 0; i <= num_steps; ++i) {
    const double t = static_cast<double>(i) / num_steps;
    const double delta_theta = normalize_angle(to.theta - from.theta);
    path_segment.emplace_back((1.0 - t) * from.x + t * to.x,
                              (1.0 - t) * from.y + t * to.y,
                              normalize_angle(from.theta + t * delta_theta));
  }
  return !checker_->is_path_in_collision(path_segment);
}

inline RRTConnectPlanner::ExtendResult
RRTConnectPlanner::extend(Tree &tree, const Config &q, double step_size,
                            double resolution) {
  const int nearest_id = find_nearest(tree, q);
  const Config &q_near = tree.nodes[nearest_id].config;
  const Config q_new = steer(q_near, q, step_size);
  if (!is_path_collision_free(q_near, q_new, resolution)) {
    return ExtendResult::Trapped;
  }
  tree.nodes.push_back({q_new, nearest_id});
  const double residual = calculate_distance(q_new, q);
  return residual < 1e-9 ? ExtendResult::Reached : ExtendResult::Advanced;
}

inline RRTConnectPlanner::ExtendResult
RRTConnectPlanner::connect(Tree &tree, const Config &q, double step_size,
                             double resolution) {
  ExtendResult er;
  do {
    er = extend(tree, q, step_size, resolution);
  } while (er == ExtendResult::Advanced);
  return er;
}

inline double RRTConnectPlanner::calculate_distance(const Config &from,
                                                      const Config &to) {
  const double dx = from.x - to.x;
  const double dy = from.y - to.y;
  const double d_theta = normalize_angle(from.theta - to.theta);
  return std::sqrt(dx * dx + dy * dy + 0.1 * (d_theta * d_theta));
}

inline double RRTConnectPlanner::normalize_angle(double angle) {
  return angle - 2.0 * M_PI * std::floor((angle + M_PI) / (2.0 * M_PI));
}

inline std::vector<Config>
RRTConnectPlanner::reconstruct(const Tree &tree, int node_id) {
  std::vector<Config> path;
  int id = node_id;
  while (id != -1) {
    path.push_back(tree.nodes[id].config);
    id = tree.nodes[id].parent_id;
  }
  return path;
}

} // namespace object_planner
