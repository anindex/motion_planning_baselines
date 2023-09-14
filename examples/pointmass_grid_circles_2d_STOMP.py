import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite
from mp_baselines.planners.stomp import STOMP
from torch_robotics.environments.env_grid_circles_2d import EnvGridCircles2D
from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    seed = 0
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvGridCircles2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.005,
        tensor_args=tensor_args
    )

    robot = RobotPointMass(
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-0.81, -0.81], [0.95, 0.95]], **tensor_args),  # workspace limits
        obstacle_cutoff_margin=0.005,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    start_state = torch.tensor([-0.8, -0.8], **tensor_args)
    goal_state = torch.tensor([0.8, 0.8], **tensor_args)

    multi_goal_states = goal_state.unsqueeze(0)

    n_support_points = 64
    dt = 0.04

    # Construct cost function
    sigma_coll = 1e-3
    cost_collisions = []
    for collision_field in task.get_collision_fields():
        cost_collisions.append(
            CostCollision(
                robot, n_support_points,
                field=collision_field,
                sigma_coll=sigma_coll,
                tensor_args=tensor_args
            )
        )

    cost_func_list = [*cost_collisions]
    cost_composite = CostComposite(
        robot, n_support_points, cost_func_list,
        tensor_args=tensor_args
    )

    num_particles_per_goal = 4
    opt_iters = 20

    planner_params = dict(
        n_dof=robot.q_dim,
        n_support_points=n_support_points,
        num_particles_per_goal=num_particles_per_goal,
        opt_iters=1,  # Keep this 1 for visualization
        num_samples=30,
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
    planner = STOMP(**planner_params)

    # Optimize
    trajs_0 = planner.get_traj()
    trajs_iters = torch.empty((opt_iters + 1, *trajs_0.shape), **tensor_args)
    trajs_iters[0] = trajs_0
    with TimerCUDA() as t:
        for i in range(opt_iters):
            trajs = planner.optimize(debug=True)
            trajs_iters[i+1] = trajs
    print(f'Optimization time: {t.elapsed:.3f} sec')

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

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
