#pragma once

#include "batched_collision_checker.h"
#include <random>
#include <vector>

namespace object_planner {

class PathSmoother {
public:
  /**
   * @brief Construct a new Path Smoother object.
   * @param checker A pointer to a pre-configured BatchedCollisionChecker.
   */
  PathSmoother(const BatchedCollisionChecker *checker);

  /**
   * @brief Smooths a given path using random shortcutting.
   * @param path The initial path to smooth.
   * @param iterations The number of smoothing attempts to make.
   * @return A new, potentially shorter and smoother path.
   */
  std::vector<Config> smooth(std::vector<Config> path, int iterations);

private:
  bool is_path_collision_free(const Config &from, const Config &to,
                              double resolution);

  const BatchedCollisionChecker *checker_;
  std::mt19937 random_engine_;
};

} // namespace object_planner