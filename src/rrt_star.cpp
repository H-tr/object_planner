#include "rrt_star.h"
#include <algorithm>
#include <cmath>

namespace object_planner {

RRTStarPlanner::RRTStarPlanner(const BatchedCollisionChecker *checker,
                               Config bounds_min, Config bounds_max)
    : checker_(checker), bounds_min_(bounds_min), bounds_max_(bounds_max),
      random_engine_(std::random_device{}()), uniform_dist_(0.0, 1.0) {}

std::vector<Config> RRTStarPlanner::plan(const Config &start,
                                         const Config &goal,
                                         const PlanParams &params) {
  std::vector<RRTNode> nodes;
  nodes.emplace_back(RRTNode{start, -1, 0.0});

  for (int i = 0; i < params.max_iterations; ++i) {
    // 1. Sample a random configuration
    Config sampled_config = (uniform_dist_(random_engine_) < params.goal_bias)
                                ? goal
                                : sample_random_config();

    // 2. Find the nearest node in the tree
    int nearest_node_id = find_nearest_node(nodes, sampled_config);
    const RRTNode &nearest_node = nodes[nearest_node_id];

    // 3. Steer from the nearest node towards the sample
    Config new_config =
        steer(nearest_node.config, sampled_config, params.step_size);

    // 4. Check for collision on the path to the new config
    if (!is_path_collision_free(nearest_node.config, new_config,
                                params.collision_check_resolution)) {
      continue;
    }

    // 5. RRT* Core: Find neighbors and choose the best parent
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

    // 6. Add the new node to the tree
    int new_node_id = nodes.size();
    nodes.emplace_back(RRTNode{new_config, best_parent_id, min_cost});

    // 7. RRT* Core: Rewire the tree
    for (int nearby_id : nearby_node_ids) {
      if (nearby_id == best_parent_id)
        continue;

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

  // After iterations, find the node closest to the goal
  int goal_node_id = -1;
  double min_goal_dist = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < nodes.size(); ++i) {
    double dist = calculate_distance(nodes[i].config, goal);
    if (dist < min_goal_dist) {
      // Also check if path to goal is clear
      if (is_path_collision_free(nodes[i].config, goal,
                                 params.collision_check_resolution)) {
        min_goal_dist = dist;
        goal_node_id = i;
      }
    }
  }

  if (goal_node_id != -1) {
    auto path = reconstruct_path(nodes, goal_node_id);
    path.push_back(goal); // Add final goal config
    return path;
  }

  return {}; // No path found
}

// --- Helper Method Implementations ---

Config RRTStarPlanner::sample_random_config() {
  return {uniform_dist_(random_engine_) * (bounds_max_.x - bounds_min_.x) +
              bounds_min_.x,
          uniform_dist_(random_engine_) * (bounds_max_.y - bounds_min_.y) +
              bounds_min_.y,
          uniform_dist_(random_engine_) *
                  (bounds_max_.theta - bounds_min_.theta) +
              bounds_min_.theta};
}

int RRTStarPlanner::find_nearest_node(const std::vector<RRTNode> &nodes,
                                      const Config &config) {
  double min_dist = std::numeric_limits<double>::infinity();
  int nearest_node_id = -1;
  for (size_t i = 0; i < nodes.size(); ++i) {
    double dist = calculate_distance(nodes[i].config, config);
    if (dist < min_dist) {
      min_dist = dist;
      nearest_node_id = i;
    }
  }
  return nearest_node_id;
}

Config RRTStarPlanner::steer(const Config &from, const Config &to,
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

bool RRTStarPlanner::is_path_collision_free(const Config &from,
                                            const Config &to,
                                            double resolution) {
  std::vector<Config> path_segment;
  double dist = calculate_distance(from, to);
  int num_steps = static_cast<int>(dist / resolution);
  if (num_steps < 2)
    num_steps = 2;

  for (int i = 0; i <= num_steps; ++i) {
    double t = static_cast<double>(i) / num_steps;
    double delta_theta = normalize_angle(to.theta - from.theta);
    path_segment.emplace_back((1.0 - t) * from.x + t * to.x,
                              (1.0 - t) * from.y + t * to.y,
                              normalize_angle(from.theta + t * delta_theta));
  }
  return !checker_->is_path_in_collision(path_segment);
}

void RRTStarPlanner::find_nearby_nodes(const std::vector<RRTNode> &nodes,
                                       const Config &config, double radius,
                                       std::vector<int> &nearby_indices) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (calculate_distance(nodes[i].config, config) <= radius) {
      nearby_indices.push_back(i);
    }
  }
}

double RRTStarPlanner::calculate_distance(const Config &from,
                                          const Config &to) {
  double dx = from.x - to.x;
  double dy = from.y - to.y;
  double d_theta = normalize_angle(from.theta - to.theta);
  // Weighting can be important here. We'll weight rotation less.
  return std::sqrt(dx * dx + dy * dy + 0.1 * (d_theta * d_theta));
}

double RRTStarPlanner::normalize_angle(double angle) {
  return angle - 2.0 * M_PI * std::floor((angle + M_PI) / (2.0 * M_PI));
}

std::vector<Config>
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