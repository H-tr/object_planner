#pragma once

#include "batched_collision_checker.h"
#include <random>
#include <vector>

namespace object_planner {

// A node in the RRT* search tree
struct RRTNode {
  Config config;
  int parent_id = -1;
  double cost = 0.0; // Cost from the root node
};

class RRTStarPlanner {
public:
  // Parameters to control the RRT* planning process
  struct PlanParams {
    int max_iterations = 5000;
    double step_size = 0.1; // Max distance to extend the tree in one step
    double goal_bias = 0.1; // Probability of sampling the goal directly
    double neighborhood_radius = 0.5; // Radius for finding neighbors to rewire
    double collision_check_resolution =
        0.01; // Resolution for path checking (e.g., 1cm)
  };

  /**
   * @brief Construct a new RRTStarPlanner object
   * @param checker A pointer to a pre-configured BatchedCollisionChecker. The
   * planner does not own this pointer.
   * @param bounds_min The minimum (x, y, theta) values for the configuration
   * space.
   * @param bounds_max The maximum (x, y, theta) values for the configuration
   * space.
   */
  RRTStarPlanner(const BatchedCollisionChecker *checker, Config bounds_min,
                 Config bounds_max);

  /**
   * @brief Plan a path from a start to a goal configuration.
   * @param start The starting configuration.
   * @param goal The desired goal configuration.
   * @param params The planning parameters to use.
   * @return A vector of Config waypoints representing the path, or an empty
   * vector if no path is found.
   */
  std::vector<Config> plan(const Config &start, const Config &goal,
                           const PlanParams &params);

private:
  // Private helper methods for the RRT* algorithm
  Config sample_random_config();
  int find_nearest_node(const std::vector<RRTNode> &nodes,
                        const Config &config);
  Config steer(const Config &from, const Config &to, double step_size);
  bool is_path_collision_free(const Config &from, const Config &to,
                              double resolution);
  void find_nearby_nodes(const std::vector<RRTNode> &nodes,
                         const Config &config, double radius,
                         std::vector<int> &nearby_indices);
  double calculate_distance(const Config &from, const Config &to);
  double normalize_angle(double angle);
  std::vector<Config> reconstruct_path(const std::vector<RRTNode> &nodes,
                                       int goal_node_id);

  // Member variables
  const BatchedCollisionChecker *checker_;
  Config bounds_min_, bounds_max_;

  // For random number generation
  std::mt19937 random_engine_;
  std::uniform_real_distribution<double> uniform_dist_;
};

} // namespace object_planner