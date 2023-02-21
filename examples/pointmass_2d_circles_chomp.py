import time

import matplotlib.pyplot as plt
import torch

from examples.pointmass_2d_circles_rrt_star import create_grid_circles
from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.utils import elapsed_time
from robot_envs.base_envs.pointmass_env_base import PointMassEnvBase
from stoch_gpmp.costs.cost_functions import CostComposite
from torch_kinematics_tree.geometrics.utils import to_torch, to_numpy

if __name__ == "__main__":
    seed = 0
    fix_random_seed(seed)

    device = 'cuda'
    tensor_args = {'device': device, 'dtype': torch.float64}

    # -------------------------------- Environment ---------------------------------
    rows = 8
    cols = 8
    radius = 0.075
    obst_primitives_l = create_grid_circles(rows, cols, radius, tensor_args=tensor_args)

    env = PointMassEnvBase(
        obst_primitives_l=obst_primitives_l,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    traj_len = 64

    # Construct cost function
    cost_func_list = [env.compute_collision_cost]
    cost_composite = CostComposite(env.q_n_dofs, traj_len, cost_func_list)

    # q_start = env.random_coll_free_q(max_samples=1000)
    # q_goal = env.random_coll_free_q(max_samples=1000)

    q_start = to_torch([-0.75, -0.9], **tensor_args)
    q_goal = -1. * q_start

    start_state = q_start
    multi_goal_states = q_goal.unsqueeze(0)

    dt = 0.02
    num_particles_per_goal = 20

    chomp_params = dict(
        n_dof=env.q_n_dofs,
        traj_len=traj_len,
        num_particles_per_goal=num_particles_per_goal,
        opt_iters=1,  # Keep this 1 for visualization
        dt=dt,
        start_state=start_state,
        cost=cost_composite,
        step_size=0.01,
        grad_clip=.1,
        multi_goal_states=multi_goal_states,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=5.,
        pos_only=False,
        tensor_args=tensor_args,
    )
    planner = CHOMP(**chomp_params)

    # ---------------------------------------------------------------------------
    # Optimize
    opt_iters = 1000

    traj_history = []
    time_start = time.time()
    for i in range(opt_iters + 1):
        trajectories = planner.optimize()
        traj_history.append(trajectories)
    print(f'Planning time: {elapsed_time(time_start):.2f}')

    # ---------------------------------------------------------------------------
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    env.render(ax)
    # plot path
    for traj in trajectories:
        traj = to_numpy(traj)
        traj_pos = env.get_q_position(traj)
        ax.plot(traj_pos[:, 0], traj_pos[:, 1])
        ax.scatter(traj_pos[0][0], traj_pos[0][1], color='green', s=50)
        ax.scatter(traj_pos[-1][0], traj_pos[-1][1], color='red', s=50)
    plt.show()
