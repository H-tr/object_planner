"""Plan with BIT*, the cost-optimising planner.

Unlike RRT-Connect (see run_rrtc_planner.py), BIT* keeps improving the path it has
for as long as you give it batches. This demo sweeps the batch budget so you
can watch the cost come down, then draws the best path it found.

    python examples/run_bit_star.py            # plan and visualize
    python examples/run_bit_star.py --no-viz   # plan only, no Open3D window

The cost being minimised lives in include/object_planner/cost_function.hpp.
Its `state_cost` stub returns 0, so out of the box BIT* minimises path length.
Implement the stub and list the poses it depends on in `cost_context` below.
"""

import random
import sys
import time

import numpy as np
import object_planner_py as opp

# --- CONFIGURATION SECTION ---
MAP_X_BOUNDS = (-1.5, 1.5)
MAP_Y_BOUNDS = (-1.5, 1.5)
MAP_THETA_BOUNDS = (-np.pi, np.pi)
NUM_OBSTACLES = 30
OBSTACLE_HEIGHT = 0.2
POINT_INFLATION = 0.015
SAMPLES_PER_BATCH = 150
# Each entry is an independent run with that many batches; more batches means
# more samples, so the cost trends down. Individual runs still vary.
BATCH_SCHEDULE = (1, 2, 4, 8, 16)
# --- END CONFIGURATION ---


def se2_distance(a, b):
    """Mirrors CostFunction::distance: the metric the planner plans in.

    Because state_cost is non-negative, this is a hard lower bound on the cost
    of any path from `a` to `b` -- handy for reading the optimality gap.
    """
    dtheta = (b.theta - a.theta + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + 0.1 * dtheta**2)


def build_scene():
    w, d, h = 0.1, 0.1, 0.02
    object_points = np.array(
        [
            [x, y, z]
            for x in np.linspace(-w / 2, w / 2, 5)
            for y in np.linspace(-d / 2, d / 2, 5)
            for z in np.linspace(0, h, 2)
        ]
    )
    all_obstacle_points = []
    for _ in range(NUM_OBSTACLES):
        pos_x = random.uniform(*MAP_X_BOUNDS)
        pos_y = random.uniform(*MAP_Y_BOUNDS)
        size_x = random.uniform(0.05, 0.4)
        size_y = random.uniform(0.05, 0.4)
        x_pts = int(size_x / 0.02) if size_x > 0.02 else 2
        y_pts = int(size_y / 0.02) if size_y > 0.02 else 2
        x_c = np.linspace(pos_x - size_x / 2, pos_x + size_x / 2, x_pts)
        y_c = np.linspace(pos_y - size_y / 2, pos_y + size_y / 2, y_pts)
        z_c = np.linspace(0, OBSTACLE_HEIGHT, 5)
        X, Y, Z = np.meshgrid(x_c, y_c, z_c)
        all_obstacle_points.append(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T)
    return object_points, np.vstack(all_obstacle_points)


def generate_valid_random_config(planner, x_bounds, y_bounds, theta_bounds):
    for _ in range(1000):
        config = opp.Config(
            random.uniform(*x_bounds),
            random.uniform(*y_bounds),
            random.uniform(*theta_bounds),
        )
        if not planner.is_config_in_collision(config):
            return config
    raise RuntimeError("Failed to find a valid random config after 1000 tries.")


def visualize(object_points, obstacle_points, path, start_config, goal_config):
    import open3d as o3d

    def object_at(config, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        T = np.eye(4)
        T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, config.theta))
        T[0, 3], T[1, 3] = config.x, config.y
        pcd.transform(T)
        pcd.paint_uniform_color(color)
        return pcd

    obstacle_pcd = o3d.geometry.PointCloud()
    obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points)
    obstacle_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    geometries = [
        obstacle_pcd,
        object_at(start_config, [0.0, 0.8, 0.2]),
        object_at(goal_config, [0.0, 0.2, 0.8]),
    ]

    path_points_3d = [[c.x, c.y, 0.01] for c in path]
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(path_points_3d),
        lines=o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(path_points_3d) - 1)]
        ),
    )
    lineset.colors = o3d.utility.Vector3dVector(
        [[1, 0, 0] for _ in range(len(lineset.lines))]
    )
    geometries.append(lineset)

    for point in path_points_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(point)
        sphere.paint_uniform_color([0.6, 0.2, 0.8])
        geometries.append(sphere)

    # A few ghosts of the object along the way, to show the orientation.
    num_ghosts = 4
    if len(path) > num_ghosts + 1:
        interior = path[1:-1]
        step = max(1, len(interior) // num_ghosts)
        for i in range(0, len(interior), step)[:num_ghosts]:
            geometries.append(object_at(interior[i], [0.9, 0.7, 0.1]))

    table = o3d.geometry.TriangleMesh.create_box(
        width=MAP_X_BOUNDS[1] - MAP_X_BOUNDS[0],
        height=MAP_Y_BOUNDS[1] - MAP_Y_BOUNDS[0],
        depth=0.005,
    )
    table.translate([MAP_X_BOUNDS[0], MAP_Y_BOUNDS[0], -0.005])
    table.paint_uniform_color([0.8, 0.7, 0.6])
    geometries.append(table)
    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    )

    print("\n--- Displaying Visualization ---")
    o3d.visualization.draw_geometries(geometries)


def main():
    show_viz = "--no-viz" not in sys.argv

    print("--- Setting up Large, Cluttered Scene for Cross-Map Planning ---")
    object_points, obstacle_points = build_scene()
    print(f"Object: {len(object_points)} pts. Obstacles: {len(obstacle_points)} pts.")

    print("\n--- Initializing Planner ---")
    planner = opp.Planner(
        object_points=object_points,
        obstacle_points=obstacle_points,
        x_bounds=MAP_X_BOUNDS,
        y_bounds=MAP_Y_BOUNDS,
        theta_bounds=MAP_THETA_BOUNDS,
        point_inflation=POINT_INFLATION,
        # The (x, y, theta) dependencies CostFunction::state_cost reads. The
        # stub ignores them, so leaving this empty changes nothing today.
        cost_context=[],
    )

    print("--- Generating Start and Goal on Opposite Sides of the Map ---")
    zone_width = (MAP_X_BOUNDS[1] - MAP_X_BOUNDS[0]) * 0.2
    start_config = generate_valid_random_config(
        planner,
        (MAP_X_BOUNDS[0], MAP_X_BOUNDS[0] + zone_width),
        MAP_Y_BOUNDS,
        MAP_THETA_BOUNDS,
    )
    goal_config = generate_valid_random_config(
        planner,
        (MAP_X_BOUNDS[1] - zone_width, MAP_X_BOUNDS[1]),
        MAP_Y_BOUNDS,
        MAP_THETA_BOUNDS,
    )
    lower_bound = se2_distance(start_config, goal_config)

    print(f"\n--- Planning from {start_config} to {goal_config} ---")
    print(f"No path can cost less than {lower_bound:.4f} (straight-line bound).")
    print("Each row is a fresh run, so the cost wobbles; the trend is what matters.\n")
    print(
        f"{'batches':>8} {'waypoints':>10} {'cost':>10} {'over bound':>12} {'seconds':>9}"
    )

    params = opp.BITStarParams()
    params.samples_per_batch = SAMPLES_PER_BATCH
    best_path, best_cost = [], float("inf")
    for batches in BATCH_SCHEDULE:
        params.max_batches = batches
        start_time = time.time()
        path = planner.plan_bit_star(
            start=start_config, goal=goal_config, plan_params=params
        )
        elapsed = time.time() - start_time
        cost = planner.bit_star_solution_cost()
        if not path:
            print(f"{batches:>8} {'-':>10} {'no path':>10} {'-':>12} {elapsed:>9.4f}")
            continue
        excess = 100.0 * (cost / lower_bound - 1.0)
        print(
            f"{batches:>8} {len(path):>10} {cost:>10.4f} {excess:>11.1f}% {elapsed:>9.4f}"
        )
        if cost < best_cost:
            best_path, best_cost = path, cost

    if not best_path:
        print("\n--- Failure ---")
        print("BIT* found no path. The random scene might be impossible; try again.")
        return

    print(f"\nBest path: {len(best_path)} waypoints, cost {best_cost:.4f}.")
    if show_viz:
        visualize(object_points, obstacle_points, best_path, start_config, goal_config)
    else:
        print("Skipping visualization (--no-viz).")


if __name__ == "__main__":
    main()
