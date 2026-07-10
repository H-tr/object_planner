#pragma once

#include "batched_collision_checker.hpp"
#include "cost_function.hpp"
#include "data_structures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace object_planner {

// Batch Informed Trees (BIT*), after Gammell et al. 2015 and the classic
// two-queue ompl::geometric::BITstar (OMPL 1.4.2; OMPL's current BITstar
// is really Advanced BIT*, which drops the vertex queue).
//
// Each batch draws `samples_per_batch` configurations from the informed
// set -- the states that could still lie on a path cheaper than the
// incumbent -- and sweeps the implicit random geometric graph over every
// live state best-first, ordered by an admissible estimate of the
// solution cost through each edge. Edges are collision-checked only when
// they reach the front of the queue, so the expensive check is paid only
// for the edges the search actually wants. Existing vertices are rewired
// whenever a cheaper way into them turns up. The solution cost decreases
// monotonically, batch after batch.
//
// Unlike RRTConnectPlanner this optimises: it returns the cheapest path
// found under `CostFunction`, not the first feasible one.
class BITStarPlanner {
public:
  struct PlanParams {
    int max_batches = 20;
    int samples_per_batch = 100;
    double collision_check_resolution = 0.01;
  };

  // `cost_context` is the list of (x, y, theta) dependencies the cost
  // function needs. It is fixed here, when the planner is set up, and is
  // read by CostFunction::state_cost throughout planning.
  inline BITStarPlanner(const BatchedCollisionChecker *checker,
                         Config bounds_min, Config bounds_max,
                         std::vector<Config> cost_context = {});

  // Cheapest path found within the batch budget; empty if none was.
  inline std::vector<Config> plan(const Config &start, const Config &goal,
                                    const PlanParams &params);

  // Total CostFunction cost of the path plan() last returned; infinity if
  // that call found none.
  inline double solution_cost() const;

private:
  static constexpr int kStartId = 0;
  static constexpr int kGoalId = 1;
  static constexpr int kDimensions = 3;      // (x, y, theta)
  static constexpr double kRewireFactor = 1.1;  // OMPL's default
  static constexpr int kMaxSampleAttempts = 100;
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();

  // Every state -- tree vertex and free sample alike -- lives in one
  // array and one neighbour pool, as in OMPL's ImplicitGraph: joining
  // the tree flips `in_tree` rather than moving the state elsewhere.
  struct State {
    Config config;
    double g_hat = 0.0;   // admissible cost-to-come from the start
    double h_hat = 0.0;   // admissible cost-to-go to the goal
    double g = kInfinity; // g_T: true cost-to-come through the tree
    double edge_cost = 0.0; // true cost of the edge from `parent`
    std::vector<int> children;
    int parent = -1;
    bool in_tree = false;
    bool is_new = true;                // free sample drawn this batch
    bool expanded_to_samples = false;  // has queued edges to nearby samples
    bool expanded_to_vertices = false; // has queued rewiring edges
    bool pruned = false;
  };

  struct VertexEntry {
    double key; // g_T(v) + h_hat(v)
    int v;
    inline bool operator>(const VertexEntry &o) const { return key > o.key; }
  };

  struct EdgeEntry {
    double key; // g_T(v) + c_hat(v, x) + h_hat(x)
    int v, x;
    inline bool operator>(const EdgeEntry &o) const { return key > o.key; }
  };

  using VertexQueue = std::priority_queue<VertexEntry, std::vector<VertexEntry>,
                                            std::greater<VertexEntry>>;
  using EdgeQueue = std::priority_queue<EdgeEntry, std::vector<EdgeEntry>,
                                          std::greater<EdgeEntry>>;

  inline void start_batch(const PlanParams &params);
  inline void run_batch(const PlanParams &params);
  inline void expand_until_edge_is_better();
  inline void expand_vertex(int v);
  inline void process_edge(const EdgeEntry &e, const PlanParams &params);
  inline void connect(int v, int x, double true_cost);
  inline void update_descendants(int root);

  inline void prune();
  inline void draw_samples(int count);
  inline bool sample_informed(Config &out);
  inline Config sample_uniform();
  inline void find_k_nearest(int v);
  inline int compute_k(std::size_t num_states) const;

  inline void add_state(const Config &config);
  inline bool segment_collision_free(const Config &a, const Config &b,
                                       double resolution) const;
  inline double c_hat(int a, int b) const;
  inline double vertex_key(int v) const;
  inline double edge_key(int v, int x) const;
  inline std::vector<Config> extract_path() const;

  const BatchedCollisionChecker *checker_;
  Config bounds_min_, bounds_max_;
  CostFunction cost_;
  std::mt19937 random_engine_;
  std::uniform_real_distribution<double> uniform_dist_;

  std::vector<State> states_;
  std::vector<int> live_;      // unpruned states: tree vertices + samples
  std::vector<int> neighbors_; // scratch: result of the last k-NN query
  std::vector<std::pair<double, int>> distances_; // scratch for find_k_nearest
  VertexQueue vertex_queue_;
  EdgeQueue edge_queue_;
  double best_cost_ = kInfinity; // c_i: cost of the incumbent solution
  int k_ = 1;
};

inline BITStarPlanner::BITStarPlanner(const BatchedCollisionChecker *checker,
                                        Config bounds_min, Config bounds_max,
                                        std::vector<Config> cost_context)
    : checker_(checker), bounds_min_(bounds_min), bounds_max_(bounds_max),
      cost_(std::move(cost_context)), random_engine_(std::random_device{}()),
      uniform_dist_(0.0, 1.0) {}

inline double BITStarPlanner::solution_cost() const { return best_cost_; }

inline std::vector<Config> BITStarPlanner::plan(const Config &start,
                                                  const Config &goal,
                                                  const PlanParams &params) {
  states_.clear();
  live_.clear();
  vertex_queue_ = VertexQueue();
  edge_queue_ = EdgeQueue();
  best_cost_ = kInfinity;

  // The start roots the tree; the goal enters as a free sample and is
  // pulled into the tree by the search, exactly as in OMPL. Both are
  // seeded by hand because add_state() reads them to build heuristics.
  const double span = cost_.heuristic_cost(start, goal);
  states_.resize(2);
  live_ = {kStartId, kGoalId};
  states_[kStartId].config = start;
  states_[kStartId].h_hat = span;
  states_[kStartId].g = 0.0;
  states_[kStartId].in_tree = true;
  states_[kGoalId].config = goal;
  states_[kGoalId].g_hat = span;

  for (int batch = 0; batch < params.max_batches; ++batch) {
    start_batch(params);
    // Nothing left that could beat the incumbent: the search has
    // converged and further batches cannot improve it.
    if (vertex_queue_.empty()) break;
    run_batch(params);
  }
  return extract_path();
}

// Prune, draw a fresh informed batch, recompute the connection radius and
// re-queue every vertex that could still improve the solution.
inline void BITStarPlanner::start_batch(const PlanParams &params) {
  prune();
  for (int i : live_) {
    if (!states_[i].in_tree) states_[i].is_new = false;
  }
  draw_samples(params.samples_per_batch);
  // Drawing first means the first batch sizes k from the samples it is
  // about to search, matching OMPL's first-batch bootstrap.
  k_ = compute_k(live_.size());
  for (int i : live_) {
    if (states_[i].in_tree && vertex_key(i) < best_cost_) {
      vertex_queue_.push({vertex_key(i), i});
    }
  }
}

// Interleave vertex expansion and edge processing until both queues run
// dry, which is when no queued edge can improve the incumbent any more.
inline void BITStarPlanner::run_batch(const PlanParams &params) {
  for (;;) {
    expand_until_edge_is_better();
    if (edge_queue_.empty()) return;
    const EdgeEntry e = edge_queue_.top();
    edge_queue_.pop();
    process_edge(e, params);
  }
}

// Expand vertices while the most promising one is at least as promising
// as the most promising queued edge.
inline void BITStarPlanner::expand_until_edge_is_better() {
  for (;;) {
    if (vertex_queue_.empty()) return;
    const double best_edge =
        edge_queue_.empty() ? kInfinity : edge_queue_.top().key;
    if (vertex_queue_.top().key > best_edge) return;
    const int v = vertex_queue_.top().v;
    vertex_queue_.pop();
    // Rewiring may have lowered g_T(v) since this entry was queued, so
    // re-test against the live value rather than the stored key.
    if (!states_[v].pruned && vertex_key(v) < best_cost_) expand_vertex(v);
  }
}

// Queue the edges out of `v` that could still lie on a better solution:
// to nearby free samples (growing the tree) and, the first time `v` is
// expanded, to nearby tree vertices (rewiring them).
inline void BITStarPlanner::expand_vertex(int v) {
  find_k_nearest(v);
  const bool first_sample_expansion = !states_[v].expanded_to_samples;
  const bool rewire = !states_[v].expanded_to_vertices;

  for (int u : neighbors_) {
    if (!states_[u].in_tree) {
      // A re-expanded vertex has already weighed the older samples.
      if (!first_sample_expansion && !states_[u].is_new) continue;
      if (states_[v].g_hat + c_hat(v, u) + states_[u].h_hat < best_cost_) {
        edge_queue_.push({edge_key(v, u), v, u});
      }
    } else if (rewire) {
      if (states_[u].parent == v || states_[v].parent == u) continue;
      if (states_[v].g_hat + c_hat(v, u) + states_[u].h_hat < best_cost_ &&
          states_[v].g + c_hat(v, u) < states_[u].g) {
        edge_queue_.push({edge_key(v, u), v, u});
      }
    }
  }
  states_[v].expanded_to_samples = true;
  states_[v].expanded_to_vertices = true;
}

// The three nested tests of BIT*, in OMPL's order: two heuristic gates
// that cost nothing, then the true edge cost, then the collision check.
inline void BITStarPlanner::process_edge(const EdgeEntry &e,
                                           const PlanParams &params) {
  const int v = e.v;
  const int x = e.x;
  if (states_[v].pruned || states_[x].pruned) return;

  // (a) Could a path through this edge beat the incumbent, optimistically?
  const double heuristic_edge = c_hat(v, x);
  if (states_[v].g + heuristic_edge + states_[x].h_hat >= best_cost_) return;
  // Could it lower x's cost-to-come, optimistically? Cheap necessary
  // condition for (c), and it gates the expensive work below.
  if (states_[v].g + heuristic_edge >= states_[x].g) return;

  const double true_cost = cost_.edge_cost(states_[v].config,
                                             states_[x].config,
                                             params.collision_check_resolution);
  // (b) Same as (a) but with the true edge cost, and with the admissible
  // cost-to-come g_hat(v) rather than the tree's g_T(v).
  if (states_[v].g_hat + true_cost + states_[x].h_hat >= best_cost_) return;
  if (!segment_collision_free(states_[v].config, states_[x].config,
                               params.collision_check_resolution)) {
    return;
  }
  // (c) Does it actually lower x's cost-to-come?
  if (states_[v].g + true_cost >= states_[x].g) return;
  connect(v, x, true_cost);
}

// Make `v` the parent of `x`, either adding `x` to the tree or rewiring
// it away from a costlier parent.
inline void BITStarPlanner::connect(int v, int x, double true_cost) {
  if (states_[x].in_tree) {
    std::vector<int> &siblings = states_[states_[x].parent].children;
    siblings.erase(std::find(siblings.begin(), siblings.end(), x));
  }
  states_[x].parent = v;
  states_[x].edge_cost = true_cost;
  states_[v].children.push_back(x);
  states_[x].g = states_[v].g + true_cost;

  const bool newly_connected = !states_[x].in_tree;
  states_[x].in_tree = true;
  update_descendants(x);
  if (states_[kGoalId].in_tree) best_cost_ = states_[kGoalId].g;
  // A brand new vertex still has to be expanded. The goal never clears
  // this test -- its key equals best_cost_ -- and expanding it is moot.
  if (newly_connected && vertex_key(x) < best_cost_) {
    vertex_queue_.push({vertex_key(x), x});
  }
}

// A rewired vertex is cheaper to reach, so everything below it is too.
// Cycles are impossible: connect() strictly lowers g_T(x), and every
// ancestor of x already has g_T <= g_T(x).
inline void BITStarPlanner::update_descendants(int root) {
  std::vector<int> stack{root};
  while (!stack.empty()) {
    const int u = stack.back();
    stack.pop_back();
    for (int c : states_[u].children) {
      states_[c].g = states_[u].g + states_[c].edge_cost;
      stack.push_back(c);
    }
  }
}

// Drop the free samples that cannot lie on a better solution. Tree
// vertices are kept (OMPL disconnects and recycles them; skipping that
// costs memory and a slightly larger k, never correctness).
inline void BITStarPlanner::prune() {
  if (!std::isfinite(best_cost_)) return;
  std::size_t kept = 0;
  for (std::size_t i = 0; i < live_.size(); ++i) {
    const int s = live_[i];
    if (!states_[s].in_tree &&
        states_[s].g_hat + states_[s].h_hat >= best_cost_) {
      states_[s].pruned = true;
      continue;
    }
    live_[kept++] = s;
  }
  live_.resize(kept);
}

inline void BITStarPlanner::draw_samples(int count) {
  for (int i = 0; i < count; ++i) {
    Config sample;
    if (!sample_informed(sample)) break;
    add_state(sample);
  }
}

// Rejection sampling of the informed set. The weighted, angle-wrapping
// SE(2) metric does not make an ellipse, so OMPL's direct prolate
// hyperspheroid sampler does not apply here.
inline bool BITStarPlanner::sample_informed(Config &out) {
  for (int attempt = 0; attempt < kMaxSampleAttempts; ++attempt) {
    const Config sample = sample_uniform();
    if (!std::isfinite(best_cost_)) {
      out = sample;
      return true;
    }
    const double f_hat =
        cost_.heuristic_cost(states_[kStartId].config, sample) +
        cost_.heuristic_cost(sample, states_[kGoalId].config);
    if (f_hat < best_cost_) {
      out = sample;
      return true;
    }
  }
  return false;
}

inline Config BITStarPlanner::sample_uniform() {
  return {uniform_dist_(random_engine_) * (bounds_max_.x - bounds_min_.x) +
              bounds_min_.x,
          uniform_dist_(random_engine_) * (bounds_max_.y - bounds_min_.y) +
              bounds_min_.y,
          uniform_dist_(random_engine_) *
                  (bounds_max_.theta - bounds_min_.theta) +
              bounds_min_.theta};
}

// One k-nearest query over every live state, as in OMPL: the caller
// splits the result into free samples and tree vertices.
inline void BITStarPlanner::find_k_nearest(int v) {
  distances_.clear();
  for (int i : live_) {
    if (i == v) continue;
    distances_.emplace_back(
        CostFunction::distance(states_[v].config, states_[i].config), i);
  }
  const std::size_t k = std::min(static_cast<std::size_t>(k_),
                                   distances_.size());
  std::nth_element(distances_.begin(), distances_.begin() + k,
                   distances_.end(),
                   [](const std::pair<double, int> &a,
                      const std::pair<double, int> &b) {
                     return a.first < b.first;
                   });
  neighbors_.clear();
  for (std::size_t i = 0; i < k; ++i) neighbors_.push_back(distances_[i].second);
}

// RGG connection constant for a k-nearest strategy: k_rgg = e + e/d
// (OMPL's ImplicitGraph::calculateMinimumRggK).
inline int BITStarPlanner::compute_k(std::size_t num_states) const {
  if (num_states < 2) return 1;
  const double k_rgg = M_E + M_E / static_cast<double>(kDimensions);
  const double k = std::ceil(kRewireFactor * k_rgg *
                              std::log(static_cast<double>(num_states)));
  return std::max(1, static_cast<int>(k));
}

// Only ever called for freshly drawn samples; plan() seeds the start and
// the goal directly.
inline void BITStarPlanner::add_state(const Config &config) {
  State state;
  state.config = config;
  state.g_hat = cost_.heuristic_cost(states_[kStartId].config, config);
  state.h_hat = cost_.heuristic_cost(config, states_[kGoalId].config);
  states_.push_back(std::move(state));
  live_.push_back(static_cast<int>(states_.size()) - 1);
}

inline bool BITStarPlanner::segment_collision_free(const Config &a,
                                                     const Config &b,
                                                     double resolution) const {
  std::vector<Config> segment;
  const double d = CostFunction::distance(a, b);
  int num_steps = static_cast<int>(d / resolution);
  if (num_steps < 2) num_steps = 2;
  for (int i = 0; i <= num_steps; ++i) {
    const double t = static_cast<double>(i) / num_steps;
    segment.push_back(CostFunction::interpolate(a, b, t));
  }
  return !checker_->is_path_in_collision(segment);
}

inline double BITStarPlanner::c_hat(int a, int b) const {
  return cost_.heuristic_cost(states_[a].config, states_[b].config);
}

inline double BITStarPlanner::vertex_key(int v) const {
  return states_[v].g + states_[v].h_hat;
}

inline double BITStarPlanner::edge_key(int v, int x) const {
  return states_[v].g + c_hat(v, x) + states_[x].h_hat;
}

inline std::vector<Config> BITStarPlanner::extract_path() const {
  if (!states_[kGoalId].in_tree) return {};
  std::vector<Config> path;
  for (int id = kGoalId; id != -1; id = states_[id].parent) {
    path.push_back(states_[id].config);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

} // namespace object_planner
