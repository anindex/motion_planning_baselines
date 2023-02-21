raise NotImplementedError

import time
import torch
import random
import matplotlib.pyplot as plt

from examples.pointmass_2d_circles_rrt_star import create_grid_circles
from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.utils import elapsed_time
from robot_envs.base_envs.pointmass_env_base import PointMassEnvBase
from torch_kinematics_tree.geometrics.utils import to_torch, to_numpy
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from torch_planning_objectives.fields.shape_distance_fields import MultiSphere
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite, CostGP
from mp_baselines.planners.stomp import STOMP


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

    #-------------------------------- Cost func. ---------------------------------
    traj_len = 64

    dt = 0.02

    # q_start = env.random_coll_free_q(max_samples=1000)
    # q_goal = env.random_coll_free_q(max_samples=1000)

    q_start = to_torch([-0.75, -0.9], **tensor_args)
    q_goal = -1. * q_start

    start_state = q_start
    multi_goal_states = q_goal.unsqueeze(0)

    # Construct cost function
    sigma_coll = 1e-4
    sigmas_prior = dict(
        sigma_start=0.001,
        sigma_gp=0.2,
    )

    start = torch.cat([start_state, torch.zeros_like(start_state)], dim=-1)
    cost_prior = CostGP(
        env.q_n_dofs, traj_len, start, dt,
        sigmas_prior, tensor_args
    )
    cost_func_list = [
        # cost_prior,
                      env.compute_collision_cost]
    cost_composite = CostComposite(env.q_n_dofs, traj_len, cost_func_list)

    # -------------------------------- Planner ---------------------------------
    num_particles_per_goal = 10
    num_samples = 30

    stomp_params = dict(
        n_dof=env.q_n_dofs,
        traj_len=traj_len,
        num_particles_per_goal=num_particles_per_goal,
        num_samples=num_samples,
        opt_iters=1,  # Keep this 1 for visualization
        dt=dt,
        start_state=start_state,
        cost=cost_composite,
        temperature=1.,
        step_size=0.1,
        sigma_spectral=0.1,
        multi_goal_states=multi_goal_states,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=5.,
        pos_only=False,
        tensor_args=tensor_args,
    )
    planner = STOMP(**stomp_params)

    # ---------------------------------------------------------------------------
    # Optimize
    opt_iters = 400

    traj_history = []
    sample_history = []
    time_start = time.time()
    for i in range(opt_iters + 1):
        time_optimize_start = time.time()
        trajectories = planner.optimize()
        traj_history.append(trajectories)
        sample_history.append(planner.state_particles.clone())
        if i % 50 == 0:
            print(i)
            print(f'{elapsed_time(time_optimize_start):.4f}')
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

