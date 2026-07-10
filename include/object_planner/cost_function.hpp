#pragma once

#include "data_structures.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace object_planner {

// The cost model BITStarPlanner optimises.
//
// A path is charged the line integral of the cost *density*
// `1 + state_cost` along it, so an edge costs the distance it covers
// plus whatever extra `state_cost` charges for the configurations it
// passes through. The default stub returns 0, which reduces the problem
// to weighted-SE(2) shortest path -- the planner is therefore runnable
// and testable before the real cost exists.
//
// `state_cost` is the only function meant to be filled in. Its input is
// one configuration; everything else it needs comes from `context()`,
// the list of (x, y, theta) dependencies handed to the planner when it
// is set up.
class CostFunction {
public:
  inline explicit CostFunction(std::vector<Config> context = {});

  inline const std::vector<Config> &context() const;

  // -------------------------------------------------------------------
  // TODO(cost): the only function to implement.
  //
  // Extra cost charged for occupying configuration `q`, on top of the
  // distance travelled. Read `context()` for the rest of the inputs.
  //
  // Must be finite and non-negative. BIT* converges to the optimum only
  // while `heuristic_cost` never overestimates `edge_cost`, and the one
  // thing that guarantees it is state_cost(q) >= 0.
  inline double state_cost(const Config &q) const;
  // -------------------------------------------------------------------

  // True cost of traversing the straight segment a -> b: the trapezoidal
  // integral of `1 + state_cost` over the same discretisation the
  // collision checker uses. Trapezoidal rather than one-sided, so both
  // endpoints carry the same weight and a rewired edge costs the same in
  // either direction.
  inline double edge_cost(const Config &a, const Config &b,
                           double resolution) const;

  // Admissible lower bound on edge_cost(a, b, *): it never calls
  // state_cost, so no cost implementation can break admissibility.
  //   edge_cost = distance + integral of state_cost >= distance.
  inline double heuristic_cost(const Config &a, const Config &b) const;

  // The weighted SE(2) metric the whole repo plans in.
  static inline double distance(const Config &a, const Config &b);
  static inline Config interpolate(const Config &a, const Config &b, double t);

private:
  static inline double normalize_angle(double angle);

  std::vector<Config> context_;
};

inline CostFunction::CostFunction(std::vector<Config> context)
    : context_(std::move(context)) {}

inline const std::vector<Config> &CostFunction::context() const {
  return context_;
}

inline double CostFunction::state_cost(const Config &q) const {
  (void)q;
  return 0.0;
}

inline double CostFunction::edge_cost(const Config &a, const Config &b,
                                        double resolution) const {
  const double length = distance(a, b);
  int num_steps = static_cast<int>(length / resolution);
  if (num_steps < 2) num_steps = 2;
  // The `1` of the density integrates to exactly `length`; only the
  // state cost needs summing.
  double integral = 0.0;
  double prev = state_cost(a);
  for (int i = 1; i <= num_steps; ++i) {
    const double t = static_cast<double>(i) / num_steps;
    const double curr = state_cost(interpolate(a, b, t));
    integral += 0.5 * (prev + curr);
    prev = curr;
  }
  const double extra = integral * (length / num_steps);
  // An admissible heuristic, a cycle-free tree and a terminating search
  // all rest on this term being non-negative, so a state_cost that
  // breaks its contract is clamped here rather than left to corrupt the
  // search or hang it. Infinity survives and reads as an impassable
  // configuration: BIT* simply discards the edge.
  return extra > 0.0 ? length + extra : length;
}

inline double CostFunction::heuristic_cost(const Config &a,
                                             const Config &b) const {
  return distance(a, b);
}

inline double CostFunction::distance(const Config &a, const Config &b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dtheta = normalize_angle(a.theta - b.theta);
  return std::sqrt(dx * dx + dy * dy + 0.1 * dtheta * dtheta);
}

inline Config CostFunction::interpolate(const Config &a, const Config &b,
                                          double t) {
  const double dtheta = normalize_angle(b.theta - a.theta);
  return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y),
          normalize_angle(a.theta + t * dtheta)};
}

inline double CostFunction::normalize_angle(double angle) {
  return angle - 2.0 * M_PI * std::floor((angle + M_PI) / (2.0 * M_PI));
}

} // namespace object_planner
