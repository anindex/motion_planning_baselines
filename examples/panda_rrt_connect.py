import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar
from mp_baselines.planners.utils import elapsed_time, extend_path
from robot_envs.base_envs.panda_env_base import PandaEnvBase
from torch_kinematics_tree.geometrics.utils import to_torch
from torch_planning_objectives.fields.primitive_distance_fields import SphereField, BoxField, InfiniteCylinderField


if __name__ == "__main__":
    seed = 9
    fix_random_seed(seed)

    tensor_args = {'device': 'cpu', 'dtype': torch.float64}

    # -------------------------------- Environment ---------------------------------
    obst_primitives_l = [
        SphereField([
            [0.6, 0.3, 0.],
            [0.5, 0.3, 0.5],
            [-0.5, 0.25, 0.6],
            [-0.6, -0.2, 0.4],
            [-0.7, 0.1, 0.0],
            [0.5, -0.45, 0.2],
            [0.6, -0.35, 0.6],
            [0.3, 0.0, 1.0],
        ],
            [
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
            ],
            tensor_args=tensor_args
        )
    ]

    env = PandaEnvBase(
        obst_primitives_l=obst_primitives_l,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    n_iters = 30000
    max_best_cost_iters = 2000
    cost_eps = 1e-2
    step_size = np.pi/80
    n_radius = np.pi/4
    n_knn = 10
    goal_prob = 0.2
    max_time = 120

    start_state = env.random_coll_free_q(max_samples=1000)
    goal_state = env.random_coll_free_q(max_samples=1000)

    rrt_params = dict(
        env=env,
        n_iters=n_iters,
        start_state=start_state,
        step_size=step_size,
        n_radius=n_radius,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    planner = RRTConnect(**rrt_params)

    # ---------------------------------------------------------------------------
    # Optimize
    start = time.time()
    traj_raw = planner.optimize(debug=True, informed=True, refill_samples_buffer=True)
    print(f"{elapsed_time(start):.3f} seconds")

    # ---------------------------------------------------------------------------
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if traj_raw is None:
        traj_raw = extend_path(env.distance_q, start_state, goal_state, max_step=np.pi/2, max_dist=torch.inf, tensor_args=tensor_args)

    env.render(ax=ax)
    env.render_trajectories(ax, [traj_raw])
    plt.show()

    # env.render_physics(traj_raw)