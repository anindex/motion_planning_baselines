import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.gpmp2 import GPMP2
from torch_robotics.environments import EnvSimple2D
from torch_robotics.environments.env_dense_2d import EnvDense2D
from torch_robotics.environments.env_dense_2d_extra_objects import EnvDense2DExtraObjects
from torch_robotics.environments.env_grid_circles_2d import EnvGridCircles2D
from torch_robotics.environments.env_narrow_passage_dense_2d import EnvNarrowPassageDense2D
from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    seed = 2
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    # env = EnvDense2D(
    #     precompute_sdf_obj_fixed=True,
    #     sdf_cell_size=0.005,
    #     tensor_args=tensor_args
    # )

    env = EnvSimple2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.005,
        tensor_args=tensor_args
    )

    # env = EnvDense2DExtraObjects(
    #     precompute_sdf_obj_fixed=True,
    #     sdf_cell_size=0.005,
    #     tensor_args=tensor_args
    # )

    # env = EnvNarrowPassageDense2D(
    #     precompute_sdf_obj_fixed=True,
    #     sdf_cell_size=0.005,
    #     tensor_args=tensor_args
    # )

    robot = RobotPointMass(
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        # ws_limits=torch.tensor([[-0.85, -0.85], [0.95, 0.95]], **tensor_args),  # workspace limits
        obstacle_buffer=0.005,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    for _ in range(100):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state = q_free[0]
        goal_state = q_free[1]

        if torch.linalg.norm(start_state - goal_state) > 1.0:
            break

    # start_state = torch.tensor([0.8956, 0.0188], **tensor_args)
    # goal_state = torch.tensor([-0.8838,  0.4582], **tensor_args)

    print(start_state)
    print(goal_state)

    # Construct planner
    num_particles_per_goal = 5

    default_params_env = env.get_gpmp2_params(robot=robot)
    n_support_points = default_params_env['n_support_points']

    duration = 5  # sec
    default_params_env['dt'] = duration / n_support_points
    dt = default_params_env['dt']

    default_params_env['sigma_gp_init'] = 1e-2

    planner_params = dict(
        **default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=num_particles_per_goal,
        start_state=start_state,
        multi_goal_states=goal_state.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    planner = GPMP2(**planner_params)

    # Optimize
    opt_iters = default_params_env['opt_iters']
    trajs_0 = planner.get_traj()
    trajs_iters = []
    trajs_iters.append(trajs_0)
    costs_previous = None
    with TimerCUDA() as t:
        for i in range(opt_iters):
            print(f'Iteration: {i}')
            trajs = planner.optimize(opt_iters=1, debug=True)
            trajs_iters.append(trajs)

            costs = planner.costs
            if i == 0:
                costs_previous = costs
                continue

            if torch.all(torch.abs((costs - costs_previous)/costs) < 0.1):
                break

            costs_previous = costs.clone()

    trajs_iters = torch.stack(trajs_iters)
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
        trajs=pos_trajs_iters[-1], start_state=start_state, goal_state=goal_state,
        render_planner=False,
    )

    planner_visualizer.animate_robot_trajectories(
        trajs=pos_trajs_iters[-1], start_state=start_state, goal_state=goal_state,
        plot_trajs=True,
        video_filepath=f'{base_file_name}-robot-traj.mp4',
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters[-1].shape[1],
        anim_time=n_support_points*dt
    )

    planner_visualizer.animate_opt_iters_robots(
        trajs=pos_trajs_iters, start_state=start_state, goal_state=goal_state,
        video_filepath=f'{base_file_name}-traj-opt-iters.mp4',
        n_frames=max((2, opt_iters//10)),
        anim_time=5
    )

    plt.show()
