import time

import matplotlib.pyplot as plt
import torch

from examples.pointmass_3d_circles_rrtstar import create_grid_circles
from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.utils import elapsed_time
from robot_envs.base_envs.pointmass_env_base import PointMassEnvBase
from stoch_gpmp.costs.cost_functions import CostComposite
from torch_kinematics_tree.geometrics.utils import to_torch, to_numpy
from torch_planning_objectives.fields.primitive_distance_fields import SphereField

if __name__ == "__main__":
    seed = 0
    fix_random_seed(seed)

    device = 'cuda'
    tensor_args = {'device': device, 'dtype': torch.float64}

    # -------------------------------- Environment ---------------------------------
    rows = 5
    cols = 5
    heights = 5
    radius = 0.075
    obst_primitives_l = create_grid_circles(rows, cols, heights, radius, tensor_args=tensor_args)

    env = PointMassEnvBase(
        q_min=(-1, -1, -1),
        q_max=(1, 1, 1),
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

    q_start = torch.tensor([-0.8, -0.8, -0.8], **tensor_args)
    q_goal = -q_start

    start_state = q_start
    multi_goal_states = q_goal.unsqueeze(0)

    dt = 0.02
    num_particles_per_goal = 10

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
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    env.render(ax)
    # plot path
    for traj in trajectories:
        traj = to_numpy(traj)
        traj_pos = env.get_q_position(traj)
        ax.plot3D(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2])
        ax.scatter3D(traj_pos[0][0], traj_pos[0][1], traj_pos[0][2], color='green', s=50)
        ax.scatter3D(traj_pos[-1][0], traj_pos[-1][1], traj_pos[-1][2], color='red', s=50)
    plt.show()

