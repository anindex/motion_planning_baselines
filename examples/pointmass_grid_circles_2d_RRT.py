import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar, InfRRTStar
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.env_dense_2d import EnvDense2D
from torch_robotics.environments.env_grid_circles_2d import EnvGridCircles2D
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.tasks.tasks import PlanningTask
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

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvGridCircles2D(
        tensor_args=tensor_args
    )

    robot = RobotPointMass(
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        # ws_limits=torch.tensor([[-0.85, -0.85], [0.95, 0.95]], **tensor_args),  # workspace limits
        # use_occupancy_map=True,  # whether to create and evaluate collisions on an occupancy map
        use_occupancy_map=False,
        cell_size=0.01,
        obstacle_cutoff_margin=0.005,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    start_state = torch.tensor([-0.8, -0.8], **tensor_args)
    goal_state = torch.tensor([0.8, 0.8], **tensor_args)

    n_iters = 30000
    step_size = 0.01
    n_radius = 0.1
    max_time = 60.

    if planner == 'rrt-connect':
        rrt_connect_params = dict(
            task=task,
            n_iters=n_iters,
            start_state_pos=start_state,
            step_size=step_size,
            n_radius=n_radius,
            max_time=max_time,
            goal_state_pos=goal_state,
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
            start_state_pos=start_state,
            step_size=step_size,
            n_radius=n_radius,
            n_knn=n_knn,
            max_time=max_time,
            goal_prob=goal_prob,
            goal_state_pos=goal_state,
            tensor_args=tensor_args,
        )

        planner = InfRRTStar(**rrt_star_params)
    else:
        raise NotImplementedError

    # Optimize
    with TimerCUDA() as t:
        traj = planner.optimize(debug=True, refill_samples_buffer=True)
    print(f'Optimization time: {t.elapsed:.3f} sec')

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    base_file_name = Path(os.path.basename(__file__)).stem

    traj = traj.unsqueeze(0)  # batch dimension for interface

    pos_trajs_iters = robot.get_position(traj)

    planner_visualizer.plot_joint_space_state_trajectories(
        trajs=traj,
        pos_start_state=start_state, pos_goal_state=goal_state,
        vel_start_state=torch.zeros_like(start_state), vel_goal_state=torch.zeros_like(goal_state),
    )

    planner_visualizer.render_robot_trajectories(
        trajs=pos_trajs_iters, start_state=start_state, goal_state=goal_state,
        render_planner=True,
    )

    planner_visualizer.animate_robot_trajectories(
        trajs=pos_trajs_iters, start_state=start_state, goal_state=goal_state,
        plot_trajs=True,
        video_filepath=f'{base_file_name}-robot-traj.mp4',
        # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
        n_frames=pos_trajs_iters.shape[1],
        anim_time=5
    )

    plt.show()
