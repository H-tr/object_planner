#pragma once

#include "batched_collision_checker.hpp"
#include "data_structures.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace object_planner {

struct RRTNode {
  Config config;
  int parent_id = -1;
  double cost = 0.0;
};

class RRTStarPlanner {
public:
  struct PlanParams {
    int max_iterations = 5000;
    double step_size = 0.1;
    double goal_bias = 0.1;
    double neighborhood_radius = 0.5;
    double collision_check_resolution = 0.01;
  };

  inline RRTStarPlanner(const BatchedCollisionChecker *checker,
                         Config bounds_min, Config bounds_max);

  inline std::vector<Config> plan(const Config &start, const Config &goal,
                                    const PlanParams &params);

private:
  inline Config sample_random_config();
  inline int find_nearest_node(const std::vector<RRTNode> &nodes,
                                 const Config &config);
  inline Config steer(const Config &from, const Config &to, double step_size);
  inline bool is_path_collision_free(const Config &from, const Config &to,
                                       double resolution);
  inline void find_nearby_nodes(const std::vector<RRTNode> &nodes,
                                  const Config &config, double radius,
                                  std::vector<int> &nearby_indices);
  inline double calculate_distance(const Config &from, const Config &to);
  inline double normalize_angle(double angle);
  inline std::vector<Config> reconstruct_path(const std::vector<RRTNode> &nodes,
                                                int goal_node_id);

  const BatchedCollisionChecker *checker_;
  Config bounds_min_, bounds_max_;
  std::mt19937 random_engine_;
  std::uniform_real_distribution<double> uniform_dist_;
};

inline RRTStarPlanner::RRTStarPlanner(const BatchedCollisionChecker *checker,
                                       Config bounds_min, Config bounds_max)
    : checker_(checker), bounds_min_(bounds_min), bounds_max_(bounds_max),
      random_engine_(std::random_device{}()), uniform_dist_(0.0, 1.0) {}

inline std::vector<Config> RRTStarPlanner::plan(const Config &start,
                                                  const Config &goal,
                                                  const PlanParams &params) {
  std::vector<RRTNode> nodes;
  nodes.emplace_back(RRTNode{start, -1, 0.0});

  for (int i = 0; i < params.max_iterations; ++i) {
    Config sampled_config = (uniform_dist_(random_engine_) < params.goal_bias)
                                ? goal
                                : sample_random_config();
    int nearest_node_id = find_nearest_node(nodes, sampled_config);
    const RRTNode &nearest_node = nodes[nearest_node_id];
    Config new_config =
        steer(nearest_node.config, sampled_config, params.step_size);
    if (!is_path_collision_free(nearest_node.config, new_config,
                                params.collision_check_resolution)) {
      continue;
    }
    std::vector<int> nearby_node_ids;
    find_nearby_nodes(nodes, new_config, params.neighborhood_radius,
                      nearby_node_ids);
    int best_parent_id = nearest_node_id;
    double min_cost =
        nearest_node.cost + calculate_distance(nearest_node.config, new_config);
    for (int nearby_id : nearby_node_ids) {
      const RRTNode &nearby_node = nodes[nearby_id];
      if (is_path_collision_free(nearby_node.config, new_config,
                                 params.collision_check_resolution)) {
        double current_cost =
            nearby_node.cost +
            calculate_distance(nearby_node.config, new_config);
        if (current_cost < min_cost) {
          min_cost = current_cost;
          best_parent_id = nearby_id;
        }
      }
    }
    int new_node_id = static_cast<int>(nodes.size());
    nodes.emplace_back(RRTNode{new_config, best_parent_id, min_cost});
    for (int nearby_id : nearby_node_ids) {
      if (nearby_id == best_parent_id) continue;
      RRTNode &nearby_node = nodes[nearby_id];
      double cost_via_new_node =
          min_cost + calculate_distance(new_config, nearby_node.config);
      if (cost_via_new_node < nearby_node.cost &&
          is_path_collision_free(new_config, nearby_node.config,
                                 params.collision_check_resolution)) {
        nearby_node.parent_id = new_node_id;
        nearby_node.cost = cost_via_new_node;
      }
    }
  }

  int goal_node_id = -1;
  double min_goal_dist = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    double dist = calculate_distance(nodes[i].config, goal);
    if (dist < min_goal_dist) {
      if (is_path_collision_free(nodes[i].config, goal,
                                 params.collision_check_resolution)) {
        min_goal_dist = dist;
        goal_node_id = static_cast<int>(i);
      }
    }
  }
  if (goal_node_id != -1) {
    auto path = reconstruct_path(nodes, goal_node_id);
    path.push_back(goal);
    return path;
  }
  return {};
}

inline Config RRTStarPlanner::sample_random_config() {
  return {uniform_dist_(random_engine_) * (bounds_max_.x - bounds_min_.x) +
              bounds_min_.x,
          uniform_dist_(random_engine_) * (bounds_max_.y - bounds_min_.y) +
              bounds_min_.y,
          uniform_dist_(random_engine_) *
                  (bounds_max_.theta - bounds_min_.theta) +
              bounds_min_.theta};
}

inline int RRTStarPlanner::find_nearest_node(const std::vector<RRTNode> &nodes,
                                              const Config &config) {
  double min_dist = std::numeric_limits<double>::infinity();
  int nearest_node_id = -1;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    double dist = calculate_distance(nodes[i].config, config);
    if (dist < min_dist) {
      min_dist = dist;
      nearest_node_id = static_cast<int>(i);
    }
  }
  return nearest_node_id;
}

inline Config RRTStarPlanner::steer(const Config &from, const Config &to,
                                      double step_size) {
  double dist = calculate_distance(from, to);
  if (dist <= step_size) {
    return to;
  }
  double ratio = step_size / dist;
  double delta_theta = normalize_angle(to.theta - from.theta);
  return {from.x + ratio * (to.x - from.x), from.y + ratio * (to.y - from.y),
          normalize_angle(from.theta + ratio * delta_theta)};
}

inline bool RRTStarPlanner::is_path_collision_free(const Config &from,
                                                     const Config &to,
                                                     double resolution) {
  std::vector<Config> path_segment;
  double dist = calculate_distance(from, to);
  int num_steps = static_cast<int>(dist / resolution);
  if (num_steps < 2) num_steps = 2;
  for (int i = 0; i <= num_steps; ++i) {
    double t = static_cast<double>(i) / num_steps;
    double delta_theta = normalize_angle(to.theta - from.theta);
    path_segment.emplace_back((1.0 - t) * from.x + t * to.x,
                              (1.0 - t) * from.y + t * to.y,
                              normalize_angle(from.theta + t * delta_theta));
  }
  return !checker_->is_path_in_collision(path_segment);
}

inline void RRTStarPlanner::find_nearby_nodes(
    const std::vector<RRTNode> &nodes, const Config &config, double radius,
    std::vector<int> &nearby_indices) {
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (calculate_distance(nodes[i].config, config) <= radius) {
      nearby_indices.push_back(static_cast<int>(i));
    }
  }
}

inline double RRTStarPlanner::calculate_distance(const Config &from,
                                                   const Config &to) {
  double dx = from.x - to.x;
  double dy = from.y - to.y;
  double d_theta = normalize_angle(from.theta - to.theta);
  return std::sqrt(dx * dx + dy * dy + 0.1 * (d_theta * d_theta));
}

inline double RRTStarPlanner::normalize_angle(double angle) {
  return angle - 2.0 * M_PI * std::floor((angle + M_PI) / (2.0 * M_PI));
}

inline std::vector<Config>
RRTStarPlanner::reconstruct_path(const std::vector<RRTNode> &nodes,
                                  int goal_node_id) {
  std::vector<Config> path;
  int current_id = goal_node_id;
  while (current_id != -1) {
    path.push_back(nodes[current_id].config);
    current_id = nodes[current_id].parent_id;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

} // namespace object_planner
