import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.pointmass_2d_circles_rrt_star import create_grid_circles
from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar
from mp_baselines.planners.utils import elapsed_time
from robot_envs.base_envs.obstacle_map_env import ObstacleMapEnv
from torch_kinematics_tree.geometrics.utils import to_numpy
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map, build_obstacle_map
from torch_planning_objectives.fields.primitive_distance_fields import SphereField


if __name__ == "__main__":
    seed = 5
    fix_random_seed(seed)

    device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float64}

    # -------------------------------- Environment ---------------------------------
    limits = torch.tensor([[-1, 1], [-1, 1]], **tensor_args)

    ## Obstacle map
    cell_size = 0.01
    map_dim = [2, 2]

    rows = 7
    cols = 7
    radius = 0.075
    obst_list = create_grid_circles(rows, cols, radius, tensor_args=tensor_args)

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        map_type='direct',
        tensor_args=tensor_args,
    )
    obst_map = build_obstacle_map(**obst_params)

    env = ObstacleMapEnv(
        name='circles',
        q_n_dofs=2,
        q_min=limits[:, 0],
        q_max=limits[:, 1],
        obstacle_map=obst_map,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    n_iters = 30000
    step_size = 0.01
    n_radius = 0.1
    max_time = 60.

    start_state = torch.tensor([-0.8, -0.8], **tensor_args)
    goal_state = torch.tensor([0.8, 0.8], **tensor_args)

    rrt_params = dict(
        env=env,
        n_iters=n_iters,
        start_state=start_state,
        step_size=step_size,
        n_radius=n_radius,
        max_time=max_time,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    planner = RRTConnect(**rrt_params)

    # ---------------------------------------------------------------------------
    # Optimize
    start = time.time()
    traj = planner.optimize(debug=True, informed=True, refill_samples_buffer=True)
    print(f"{elapsed_time(start)} seconds")

    # ---------------------------------------------------------------------------
    # Plotting
    fig, ax = plt.subplots()
    planner.render(ax)
    obst_map.plot(ax)
    if traj is not None:
        traj = to_numpy(traj)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', markersize=3)
    ax.plot(to_numpy(start_state[0]), to_numpy(start_state[1]), 'go', markersize=7)
    ax.plot(to_numpy(goal_state[0]), to_numpy(goal_state[1]), 'ro', markersize=7)
    ax.set_aspect('equal')
    plt.show()

