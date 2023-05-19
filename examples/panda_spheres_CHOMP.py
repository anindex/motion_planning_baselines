import os

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.chomp import CHOMP
from stoch_gpmp.costs.cost_functions import CostComposite
from torch_robotics.environment.env_spheres_3d import EnvSpheres3D
from torch_robotics.robot.panda_robot import PandaRobot
from torch_robotics.task.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import Timer
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    seed = 10
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvSpheres3D(tensor_args=tensor_args)

    robot = PandaRobot(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # workspace limits
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    q_free = task.random_coll_free_q(n_samples=2)
    start_state = q_free[0]
    goal_state = q_free[1]

    multi_goal_states = goal_state.unsqueeze(0)

    traj_len = 64
    dt = 0.02

    # Construct cost function
    cost_func_list = [task.compute_collision_cost]
    cost_composite = CostComposite(robot.q_dim, traj_len, cost_func_list)

    num_particles_per_goal = 3
    opt_iters = 100

    planner_params = dict(
        n_dof=robot.q_dim,
        traj_len=traj_len,
        num_particles_per_goal=num_particles_per_goal,
        opt_iters=opt_iters,  # Keep this 1 for visualization
        dt=dt,
        start_state=start_state,
        cost=cost_composite,
        step_size=0.01,
        grad_clip=.01,
        multi_goal_states=multi_goal_states,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=5.,
        pos_only=False,
        tensor_args=tensor_args,
    )
    planner = CHOMP(**planner_params)

    # Optimize
    with Timer() as t:
        traj_batch = planner.optimize(debug=True)
    print(f'Optimization time: {t.elapsed:.3f} sec')

    # -------------------------------- Visualize ---------------------------------
    traj_batch = robot.get_position(traj_batch)
    planner_visualizer = PlanningVisualizer(
        env=env,
        robot=robot,
        planner=planner
    )
    fig, ax = planner_visualizer.render_trajectory(
        traj=traj_batch[0], start_state=start_state, goal_state=goal_state, render_planner=False,
        animate=True,
        video_filepath=os.path.basename(__file__).replace('.py', '.mp4')
    )
    for traj in traj_batch[1:]:
        fig, ax = planner_visualizer.render_trajectory(
            fig=fig, ax=ax,
            traj=traj,
        )
    plt.show()
