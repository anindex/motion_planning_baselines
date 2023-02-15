import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np

from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.utils import elapsed_time
from robot_envs.base_envs.obstacle_map_env import ObstacleMapEnv
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from mp_baselines.planners.rrt_star import RRTStar


if __name__ == "__main__":
    seed = 17
    fix_random_seed(17)

    tensor_args = {'device': 'cpu', 'dtype': torch.float32}

    # -------------------------------- Environment ---------------------------------
    limits = torch.tensor([[-10, 10], [-10, 10]], **tensor_args)

    ## Obstacle map
    obst_list = []
    cell_size = 0.2
    map_dim = [20, 20]

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        map_type='direct',
        random_gen=True,
        num_obst=8,
        rand_limits=[(-7.5, 7.5), (-7.5, 7.5)],
        rand_rect_shape=[2, 2],
        rand_circle_radius=2,
        tensor_args=tensor_args,
    )

    obst_map, obst_list = generate_obstacle_map(**obst_params)

    env = ObstacleMapEnv(
        name='circles',
        q_n_dofs=2,
        q_min=limits[:, 0],
        q_max=limits[:, 1],
        obstacle_map=obst_map,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    n_iters = 5000
    max_best_cost_iters = 1000
    cost_eps = 1e-2
    step_size = 0.1
    n_radius = 2.
    n_knn = 10
    goal_prob = 0.20
    max_time = 15.
    start_state = torch.tensor([-9, -9], **tensor_args)
    goal_state = torch.tensor([9, 8], **tensor_args)

    rrt_params = dict(
        env=env,
        n_iters=n_iters,
        max_best_cost_iters=max_best_cost_iters,
        cost_eps=cost_eps,
        start_state=start_state,
        step_size=step_size,
        n_radius=n_radius,
        n_knn=n_knn,
        max_time=max_time,
        goal_prob=goal_prob,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    planner = RRTStar(**rrt_params)

    # ---------------------------------------------------------------------------
    # Optimize
    start = time.time()
    traj = planner.optimize(first_goal_return=True, debug=True, informed=True)
    print(f"{elapsed_time(start)} seconds")

    # ---------------------------------------------------------------------------
    # Plotting
    fig, ax = plt.subplots()
    planner.render(ax)
    obst_map.plot(ax)
    if traj is not None:
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', markersize=3)
    ax.plot(start_state[0], start_state[1], 'go', markersize=7)
    ax.plot(goal_state[0], goal_state[1], 'ro', markersize=7)
    ax.set_aspect('equal')
    plt.show()
