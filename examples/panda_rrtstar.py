import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.rrt_star import RRTStar
from mp_baselines.planners.utils import elapsed_time
from robot_envs.base_envs.panda_env_7d import PandaEnv7D


if __name__ == "__main__":
    seed = 7
    fix_random_seed(seed)

    tensor_args = {'device': 'cpu', 'dtype': torch.float32}

    # -------------------------------- Environment ---------------------------------
    spheres = [
        (0.5, 0.5, 0.5, 0.2),
        (-0.5, -0.5, 0.5, 0.3),
        (0.25, -0.5, 0.5, 0.25),
    ]
    boxes = []

    env = PandaEnv7D(
        obstacle_spheres=spheres,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    n_iters = 30000
    max_best_cost_iters = 2000
    cost_eps = 1e-2
    step_size = np.pi/50
    n_radius = np.pi/4
    n_knn = 10
    goal_prob = 0.1
    max_time = 60.

    start_state = env.sample_q(without_collision=True)
    goal_state = env.sample_q(without_collision=True)

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
    traj_raw = planner.optimize(debug=True, informed=True)
    print(f"{elapsed_time(start)} seconds")

    # ---------------------------------------------------------------------------
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    env.render(traj=traj_raw, ax=ax)
    plt.show()
