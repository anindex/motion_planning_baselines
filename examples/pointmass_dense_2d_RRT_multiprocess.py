import os
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mp_baselines.planners.multi_processing import MultiProcessor
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar, InfRRTStar
from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.env_grid_circles_2d import EnvGridCircles2D
from torch_robotics.environment.utils import create_grid_spheres
from torch_robotics.robot.robot_point_mass import RobotPointMass
from torch_robotics.task.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()


if __name__ == "__main__":
    planner = 'rrt-connect'
    # planner = 'rrt-star'

    seed = 0
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}
    # tensor_args = {'device': 'cpu', 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvGridCircles2D(
        tensor_args=tensor_args
    )

    robot = RobotPointMass(
        q_limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # configuration space limits
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-0.85, -0.85], [0.95, 0.95]], **tensor_args),  # workspace limits
        # use_occupancy_map=True,  # whether to create and evaluate collisions on an occupancy map
        use_occupancy_map=False,
        cell_size=0.01,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    start_state = torch.tensor([-0.8, -0.8], **tensor_args)
    goal_state = torch.tensor([-0.8, 0.8], **tensor_args)
    # start_state = torch.tensor([-0.8, -0.8], **tensor_args)
    # goal_state = torch.tensor([-0.15, -0.75], **tensor_args)

    n_iters = 30000
    step_size = 0.01
    n_radius = 0.1
    max_time = 60.

    n_trajectories = 100

    if planner == 'rrt-connect':
        rrt_connect_params = dict(
            task=task,
            n_iters=n_iters,
            start_state=start_state,
            step_size=step_size,
            n_radius=n_radius,
            max_time=max_time,
            goal_state=goal_state,
            tensor_args=tensor_args,
        )
        planner = RRTConnect(**rrt_connect_params)
    elif planner == 'rrt-star':
        max_best_cost_iters = 1000
        cost_eps = 1e-2
        n_knn = 10
        goal_prob = 0.1

        rrt_star_params = dict(
            task=task,
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
        planner = InfRRTStar(**rrt_star_params)
    else:
        raise NotImplementedError

    # Optimize in parallel
    with TimerCUDA() as t:
        sample_based_planner = MultiSampleBasedPlanner(
            planner,
            n_trajectories=n_trajectories,
            max_processes=8,
            optimize_sequentially=False
        )
        trajs_l = sample_based_planner.optimize()
    print(f'Optimization time MultiProcessing: {t.elapsed:.3f} sec')

    # Optimize sequentially
    with TimerCUDA() as t:
        sample_based_planner = MultiSampleBasedPlanner(
            planner,
            n_trajectories=n_trajectories,
            optimize_sequentially=True
        )
        trajs_l = sample_based_planner.optimize()
    print(f'Optimization time ForLoop: {t.elapsed:.3f} sec')

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    fig, ax = None, None
    for traj in trajs_l:
        if traj is None:
            continue
        traj_pos = robot.get_position(traj).unsqueeze(0)  # add batch dimension for interface
        fig, ax = planner_visualizer.render_robot_trajectories(
            fig=fig, ax=ax,
            trajs=traj_pos, start_state=start_state, goal_state=goal_state
        )

    plt.show()
