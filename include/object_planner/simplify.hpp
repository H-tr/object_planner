#pragma once

#include "batched_collision_checker.hpp"
#include "data_structures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace object_planner {

enum class SimplifyOp { Shortcut, Reduce, Perturb, BSpline };

struct ShortcutSettings {};

struct ReduceSettings {
  std::size_t max_steps = 10;
  std::size_t max_empty_steps = 5;
  double range_ratio = 0.5;
};

struct PerturbSettings {
  std::size_t max_steps = 10;
  std::size_t max_empty_steps = 5;
  std::size_t perturbation_attempts = 5;
  double range = 0.1;
};

struct BSplineSettings {
  std::size_t max_steps = 1;
  double min_change = 0.1;
  double midpoint_interpolation = 0.5;
};

struct SimplifySettings {
  std::size_t max_iterations = 5;
  std::vector<SimplifyOp> operations = {SimplifyOp::Shortcut,
                                         SimplifyOp::BSpline};
  ShortcutSettings shortcut;
  ReduceSettings reduce;
  PerturbSettings perturb;
  BSplineSettings bspline;
};

// Port of vamp/planning/simplify.hh. Each cycle of `max_iterations`
// runs the `operations` list in order; the entire simplification
// returns early if a full cycle made no progress.
class Simplifier {
public:
  inline Simplifier(const BatchedCollisionChecker *checker,
                     Config bounds_min, Config bounds_max);

  inline std::vector<Config>
  simplify(std::vector<Config> path, const SimplifySettings &settings,
           double collision_check_resolution = 0.01);

private:
  inline bool apply_shortcut(std::vector<Config> &path,
                              const ShortcutSettings &s, double res);
  inline bool apply_reduce(std::vector<Config> &path,
                            const ReduceSettings &s, double res);
  inline bool apply_perturb(std::vector<Config> &path,
                             const PerturbSettings &s, double res);
  inline bool apply_bspline(std::vector<Config> &path,
                             const BSplineSettings &s, double res);

  inline Config sample_random_in_bounds();
  inline bool segment_collision_free(const Config &a, const Config &b,
                                       double res) const;
  inline double distance(const Config &a, const Config &b) const;
  inline Config interpolate(const Config &a, const Config &b, double t) const;
  static inline double normalize_angle(double angle);

  const BatchedCollisionChecker *checker_;
  Config bounds_min_, bounds_max_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform01_;
};

inline Simplifier::Simplifier(const BatchedCollisionChecker *checker,
                                Config bounds_min, Config bounds_max)
    : checker_(checker), bounds_min_(bounds_min), bounds_max_(bounds_max),
      rng_(std::random_device{}()), uniform01_(0.0, 1.0) {}

inline std::vector<Config>
Simplifier::simplify(std::vector<Config> path,
                      const SimplifySettings &settings,
                      double collision_check_resolution) {
  if (path.size() < 3) return path;
  for (std::size_t iter = 0; iter < settings.max_iterations; ++iter) {
    bool cycle_changed = false;
    for (SimplifyOp op : settings.operations) {
      switch (op) {
      case SimplifyOp::Shortcut:
        cycle_changed |= apply_shortcut(path, settings.shortcut,
                                         collision_check_resolution);
        break;
      case SimplifyOp::Reduce:
        cycle_changed |= apply_reduce(path, settings.reduce,
                                       collision_check_resolution);
        break;
      case SimplifyOp::Perturb:
        cycle_changed |= apply_perturb(path, settings.perturb,
                                        collision_check_resolution);
        break;
      case SimplifyOp::BSpline:
        cycle_changed |= apply_bspline(path, settings.bspline,
                                        collision_check_resolution);
        break;
      }
    }
    if (!cycle_changed) break;
  }
  return path;
}

// Deterministic greedy shortcutting: for each anchor i, find the
// furthest j > i+1 that can be reached by a straight (configuration-
// space) segment and splice out everything in between.
inline bool Simplifier::apply_shortcut(std::vector<Config> &path,
                                        const ShortcutSettings & /*s*/,
                                        double res) {
  if (path.size() < 3) return false;
  bool any_change = false;
  std::size_t i = 0;
  while (i + 2 < path.size()) {
    std::size_t best_j = i + 1;
    for (std::size_t j = path.size() - 1; j > i + 1; --j) {
      if (segment_collision_free(path[i], path[j], res)) {
        best_j = j;
        break;
      }
    }
    if (best_j > i + 1) {
      path.erase(path.begin() + i + 1, path.begin() + best_j);
      any_change = true;
    }
    ++i;
  }
  return any_change;
}

inline bool Simplifier::apply_reduce(std::vector<Config> &path,
                                       const ReduceSettings &s, double res) {
  if (path.size() < 3) return false;
  bool any_change = false;
  std::size_t empty_count = 0;
  for (std::size_t step = 0; step < s.max_steps; ++step) {
    if (path.size() < 3) break;
    const std::size_t span_max = std::max<std::size_t>(
        2, static_cast<std::size_t>(static_cast<double>(path.size()) *
                                      s.range_ratio));
    std::uniform_int_distribution<std::size_t> i_dist(
        0, path.size() > 2 ? path.size() - 3 : 0);
    std::size_t i = i_dist(rng_);
    std::uniform_int_distribution<std::size_t> span_dist(2, span_max);
    std::size_t span = span_dist(rng_);
    std::size_t j = std::min(i + span, path.size() - 1);
    if (j <= i + 1) continue;
    if (segment_collision_free(path[i], path[j], res)) {
      path.erase(path.begin() + i + 1, path.begin() + j);
      empty_count = 0;
      any_change = true;
    } else {
      ++empty_count;
      if (empty_count >= s.max_empty_steps) break;
    }
  }
  return any_change;
}

inline bool Simplifier::apply_perturb(std::vector<Config> &path,
                                        const PerturbSettings &s, double res) {
  if (path.size() < 3) return false;
  bool any_change = false;
  std::size_t empty_count = 0;
  for (std::size_t step = 0; step < s.max_steps; ++step) {
    bool found = false;
    for (std::size_t attempt = 0; attempt < s.perturbation_attempts; ++attempt) {
      if (path.size() < 3) break;
      std::uniform_int_distribution<std::size_t> idx_dist(1, path.size() - 2);
      const std::size_t idx = idx_dist(rng_);
      const Config rand = sample_random_in_bounds();
      const Config cand = interpolate(path[idx], rand, s.range);
      const double cost_before = distance(path[idx - 1], path[idx]) +
                                  distance(path[idx], path[idx + 1]);
      const double cost_after = distance(path[idx - 1], cand) +
                                 distance(cand, path[idx + 1]);
      if (cost_after < cost_before &&
          segment_collision_free(path[idx - 1], cand, res) &&
          segment_collision_free(cand, path[idx + 1], res)) {
        path[idx] = cand;
        any_change = true;
        empty_count = 0;
        found = true;
        break;
      }
    }
    if (!found) {
      ++empty_count;
      if (empty_count >= s.max_empty_steps) break;
    }
  }
  return any_change;
}

// Subdivide once, then iteratively replace every other interior point
// with the midpoint of its neighbours (a single B-spline relaxation
// pass). Reports change=true iff the subdivision itself happened (it
// always does when path.size() >= 3, so the outer loop knows BSpline
// just modified the path).
inline bool Simplifier::apply_bspline(std::vector<Config> &path,
                                        const BSplineSettings &s, double res) {
  if (path.size() < 3) return false;
  std::vector<Config> subdivided;
  subdivided.reserve(path.size() * 2);
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    subdivided.push_back(path[i]);
    subdivided.push_back(interpolate(path[i], path[i + 1],
                                       s.midpoint_interpolation));
  }
  subdivided.push_back(path.back());
  path = std::move(subdivided);

  bool any_change_this_step = true;
  for (std::size_t step = 0; step < s.max_steps && any_change_this_step;
       ++step) {
    any_change_this_step = false;
    for (std::size_t i = 2; i + 1 < path.size(); i += 2) {
      const Config mid = interpolate(path[i - 1], path[i + 1], 0.5);
      const double delta = distance(path[i], mid);
      if (delta > s.min_change &&
          segment_collision_free(path[i - 1], mid, res) &&
          segment_collision_free(mid, path[i + 1], res)) {
        path[i] = mid;
        any_change_this_step = true;
      }
    }
  }
  return true;
}

inline Config Simplifier::sample_random_in_bounds() {
  return {uniform01_(rng_) * (bounds_max_.x - bounds_min_.x) + bounds_min_.x,
          uniform01_(rng_) * (bounds_max_.y - bounds_min_.y) + bounds_min_.y,
          uniform01_(rng_) * (bounds_max_.theta - bounds_min_.theta) +
              bounds_min_.theta};
}

inline bool Simplifier::segment_collision_free(const Config &a, const Config &b,
                                                  double res) const {
  std::vector<Config> seg;
  const double d = distance(a, b);
  int num_steps = static_cast<int>(d / res);
  if (num_steps < 2) num_steps = 2;
  for (int i = 0; i <= num_steps; ++i) {
    const double t = static_cast<double>(i) / num_steps;
    seg.push_back(interpolate(a, b, t));
  }
  return !checker_->is_path_in_collision(seg);
}

inline double Simplifier::distance(const Config &a, const Config &b) const {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dtheta = normalize_angle(a.theta - b.theta);
  return std::sqrt(dx * dx + dy * dy + 0.1 * dtheta * dtheta);
}

inline Config Simplifier::interpolate(const Config &a, const Config &b,
                                        double t) const {
  const double dtheta = normalize_angle(b.theta - a.theta);
  return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y),
          normalize_angle(a.theta + t * dtheta)};
}

inline double Simplifier::normalize_angle(double angle) {
  return angle - 2.0 * M_PI * std::floor((angle + M_PI) / (2.0 * M_PI));
}

} // namespace object_planner
