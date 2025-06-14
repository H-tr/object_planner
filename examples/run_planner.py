import numpy as np
import object_planner_py as opp
import time
import os
import open3d as o3d
import random


# --- Visualization Function (receives the sparse path directly) ---
def visualize_path(
    object_points,
    obstacle_points,
    path,
    start_config,
    goal_config,
    map_bounds_x,
    map_bounds_y,
):
    """
    Visualizes the sparse planning path directly from the planner.
    """
    if not path:
        print("No path to visualize.")
        return

    def create_object_at_config(p, c, clr):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        T = np.eye(4)
        T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, c.theta))
        T[0, 3] = c.x
        T[1, 3] = c.y
        pcd.transform(T)
        pcd.paint_uniform_color(clr)
        return pcd

    geometries = []
    obstacle_pcd = o3d.geometry.PointCloud()
    obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points)
    obstacle_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(obstacle_pcd)
    geometries.append(
        create_object_at_config(object_points, start_config, [0.0, 0.8, 0.2])
    )
    geometries.append(
        create_object_at_config(object_points, goal_config, [0.0, 0.2, 0.8])
    )
    path_points_3d = [[c.x, c.y, 0.01] for c in path]
    path_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(path_points_3d),
        lines=o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(path_points_3d) - 1)]
        ),
    )
    path_lineset.colors = o3d.utility.Vector3dVector(
        [[1, 0, 0] for _ in range(len(path_lineset.lines))]
    )
    geometries.append(path_lineset)

    num_ghosts = 4
    if len(path) > num_ghosts + 1:
        path_without_ends = path[1:-1]
        step = len(path_without_ends) // num_ghosts
        indices_in_subpath = range(0, len(path_without_ends), step if step > 0 else 1)[
            :num_ghosts
        ]
        for i in indices_in_subpath:
            geometries.append(
                create_object_at_config(
                    object_points, path_without_ends[i], [0.9, 0.7, 0.1]
                )
            )

    for point in path_points_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.015
        )  # Made spheres slightly larger for visibility
        sphere.translate(point)
        sphere.paint_uniform_color([0.6, 0.2, 0.8])
        geometries.append(sphere)

    map_w = map_bounds_x[1] - map_bounds_x[0]
    map_h = map_bounds_y[1] - map_bounds_y[0]
    table = o3d.geometry.TriangleMesh.create_box(width=map_w, height=map_h, depth=0.005)
    table.translate([map_bounds_x[0], map_bounds_y[0], -0.005])
    table.paint_uniform_color([0.8, 0.7, 0.6])
    geometries.append(table)
    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    )
    print("\n--- Displaying Visualization ---")
    o3d.visualization.draw_geometries(geometries)


# --- Helper function to find a valid, collision-free configuration (unchanged) ---
def generate_valid_random_config(planner, x_bounds, y_bounds, theta_bounds):
    max_tries = 1000
    for _ in range(max_tries):
        x = random.uniform(*x_bounds)
        y = random.uniform(*y_bounds)
        theta = random.uniform(*theta_bounds)
        config = opp.Config(x, y, theta)
        if not planner.is_config_in_collision(config):
            return config
    raise RuntimeError(
        f"Failed to find a valid random config in the given zone after {max_tries} tries."
    )


def main():
    # --- CONFIGURATION SECTION ---
    MAP_X_BOUNDS = (-1.5, 1.5)
    MAP_Y_BOUNDS = (-1.5, 1.5)
    MAP_THETA_BOUNDS = (-np.pi, np.pi)
    NUM_OBSTACLES = 30
    OBSTACLE_HEIGHT = 0.2
    PLANNER_ITERATIONS = 5000
    PLANNER_STEP_SIZE = 0.2
    SMOOTHING_ITERATIONS = 50
    # --- END CONFIGURATION ---

    print("--- Setting up Large, Cluttered Scene for Cross-Map Planning ---")

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
    obstacle_points = np.vstack(all_obstacle_points)
    print(f"Object: {len(object_points)} pts. Obstacles: {len(obstacle_points)} pts.")

    fk_file = "object_model.fk"
    opp.create_sphere_tree_file(object_points, fk_file)
    print("\n--- Initializing Planner ---")
    planner = opp.Planner(
        sphere_tree_file=fk_file,
        obstacle_points=obstacle_points,
        x_bounds=MAP_X_BOUNDS,
        y_bounds=MAP_Y_BOUNDS,
        theta_bounds=MAP_THETA_BOUNDS,
    )

    print("--- Generating Start and Goal on Opposite Sides of the Map ---")
    map_width = MAP_X_BOUNDS[1] - MAP_X_BOUNDS[0]
    zone_width = map_width * 0.2
    start_zone_x = (MAP_X_BOUNDS[0], MAP_X_BOUNDS[0] + zone_width)
    goal_zone_x = (MAP_X_BOUNDS[1] - zone_width, MAP_X_BOUNDS[1])
    start_config = generate_valid_random_config(
        planner, start_zone_x, MAP_Y_BOUNDS, MAP_THETA_BOUNDS
    )
    goal_config = generate_valid_random_config(
        planner, goal_zone_x, MAP_Y_BOUNDS, MAP_THETA_BOUNDS
    )

    plan_params = opp.PlanParams()
    plan_params.max_iterations = PLANNER_ITERATIONS
    plan_params.step_size = PLANNER_STEP_SIZE
    plan_params.goal_bias = 0.05
    plan_params.neighborhood_radius = 0.5
    print(f"\n--- Planning from {start_config} to {goal_config} ---")
    start_time = time.time()
    # The planner now returns the sparse path directly
    path = planner.plan(
        start=start_config,
        goal=goal_config,
        plan_params=plan_params,
        smoothing_iterations=SMOOTHING_ITERATIONS,
    )
    end_time = time.time()

    # Process and display the result
    if path and len(path) > 2:
        print(f"\n--- Results ---")
        print(
            f"SUCCESS: Path found in {end_time - start_time:.4f} seconds with {len(path)} waypoints."
        )
        # Directly visualize the sparse path
        visualize_path(
            object_points,
            obstacle_points,
            path,
            start_config,
            goal_config,
            MAP_X_BOUNDS,
            MAP_Y_BOUNDS,
        )
    else:
        print("\n--- Failure or Trivial Path ---")
        print(
            "Failed to find a non-trivial path. The random scene might be impossible. Try running again."
        )


if __name__ == "__main__":
    main()
