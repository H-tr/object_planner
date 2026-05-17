#pragma once

#include "batched_collision_checker.hpp"
#include "data_structures.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace object_planner {

class PathSmoother {
public:
  inline explicit PathSmoother(const BatchedCollisionChecker *checker);
  inline std::vector<Config> smooth(std::vector<Config> path, int iterations);

private:
  inline bool is_path_collision_free(const Config &from, const Config &to,
                                       double resolution);

  const BatchedCollisionChecker *checker_;
  std::mt19937 random_engine_;
};

inline PathSmoother::PathSmoother(const BatchedCollisionChecker *checker)
    : checker_(checker), random_engine_(std::random_device{}()) {}

inline std::vector<Config> PathSmoother::smooth(std::vector<Config> path,
                                                  int iterations) {
  if (path.size() < 3) {
    return path;
  }
  for (int i = 0; i < iterations; ++i) {
    if (path.size() < 3) break;
    std::uniform_int_distribution<std::size_t> dist(0, path.size() - 1);
    std::size_t idx1 = dist(random_engine_);
    std::size_t idx2 = dist(random_engine_);
    if (idx1 == idx2) continue;
    if (idx1 > idx2) std::swap(idx1, idx2);
    if (idx2 == idx1 + 1) continue;
    if (is_path_collision_free(path[idx1], path[idx2], 0.01)) {
      std::vector<Config> new_path;
      new_path.insert(new_path.end(), path.begin(), path.begin() + idx1 + 1);
      new_path.insert(new_path.end(), path.begin() + idx2, path.end());
      path = std::move(new_path);
    }
  }
  return path;
}

inline bool PathSmoother::is_path_collision_free(const Config &from,
                                                   const Config &to,
                                                   double resolution) {
  std::vector<Config> path_segment;
  double dx = from.x - to.x;
  double dy = from.y - to.y;
  double dist = std::sqrt(dx * dx + dy * dy);
  int num_steps = static_cast<int>(dist / resolution);
  if (num_steps < 2) num_steps = 2;
  auto normalize_angle = [](double angle) {
    return angle - 2.0 * M_PI * std::floor((angle + M_PI) / (2.0 * M_PI));
  };
  for (int i = 0; i <= num_steps; ++i) {
    double t = static_cast<double>(i) / num_steps;
    double delta_theta = normalize_angle(to.theta - from.theta);
    path_segment.emplace_back((1.0 - t) * from.x + t * to.x,
                              (1.0 - t) * from.y + t * to.y,
                              normalize_angle(from.theta + t * delta_theta));
  }
  return !checker_->is_path_in_collision(path_segment);
}

} // namespace object_planner
