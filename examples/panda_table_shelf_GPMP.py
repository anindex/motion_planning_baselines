import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.gpmp2 import GPMP2
from torch_robotics.environments.env_table_shelf import EnvTableShelf
from torch_robotics.environments.objects import GraspedObjectPandaBox
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    seed = 11
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvTableShelf(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=tensor_args
    )

    robot = RobotPanda(
        # use_self_collision_storm=True,
        # grasped_object=GraspedObjectPandaBox(tensor_args=tensor_args),
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),  # workspace limits
        obstacle_cutoff_margin=0.03,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    q_free = task.random_coll_free_q(n_samples=2)
    start_state = q_free[0]
    goal_state = q_free[1]

    # start_state = torch.tensor([ 1.9413, -0.0090,  2.3629, -0.8916,  0.2496,  3.5482, -0.7393], device='cuda:0')
    # goal_state = torch.tensor([-2.6686, -0.1020, -0.2527, -2.7064,  1.0567,  1.2865,  2.2158], device='cuda:0')

    print(start_state)
    print(goal_state)

    # Construct planner
    n_support_points = 64
    dt = 0.04
    num_particles_per_goal = 10

    default_params_env = env.get_gpmp2_params()

    planner_params = dict(
        **default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        n_support_points=n_support_points,
        num_particles_per_goal=num_particles_per_goal,
        dt=dt,
        start_state=start_state,
        multi_goal_states=goal_state.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    planner = GPMP2(**planner_params)

    # Optimize
    opt_iters = default_params_env['opt_iters']
    trajs_0 = planner.get_traj()
    trajs_iters = torch.empty((opt_iters + 1, *trajs_0.shape), **tensor_args)
    trajs_iters[0] = trajs_0
    with TimerCUDA() as t:
        for i in range(opt_iters):
            trajs = planner.optimize(opt_iters=1, debug=True)
            trajs_iters[i+1] = trajs
    print(f'Optimization time: {t.elapsed:.3f} sec')

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_iters[-1])*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_iters[-1])*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_iters[-1])}')

    base_file_name = Path(os.path.basename(__file__)).stem

    pos_trajs_iters = robot.get_position(trajs_iters)

    planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs_iters[-1],
        pos_start_state=start_state, pos_goal_state=goal_state,
        vel_start_state=torch.zeros_like(start_state), vel_goal_state=torch.zeros_like(goal_state),
    )

    planner_visualizer.animate_opt_iters_joint_space_state(
        trajs=trajs_iters,
        pos_start_state=start_state, pos_goal_state=goal_state,
        vel_start_state=torch.zeros_like(start_state), vel_goal_state=torch.zeros_like(goal_state),
        video_filepath=f'{base_file_name}-joint-space-opt-iters.mp4',
        n_frames=max((2, opt_iters // 10)),
        anim_time=5
    )

    planner_visualizer.render_robot_trajectories(
        trajs=pos_trajs_iters[-1, 0][None, ...], start_state=start_state, goal_state=goal_state,
        render_planner=False,
    )

    planner_visualizer.animate_robot_trajectories(
        trajs=pos_trajs_iters[-1, 0][None, ...], start_state=start_state, goal_state=goal_state,
        plot_trajs=False,
        video_filepath=f'{base_file_name}-robot-traj.mp4',
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters[-1].shape[1],
        anim_time=n_support_points*dt
    )

    plt.show()

