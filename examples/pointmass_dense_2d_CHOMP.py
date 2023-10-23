import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite
from torch_robotics.environments import EnvDense2D, EnvSimple2D
from torch_robotics.environments.env_grid_circles_2d import EnvGridCircles2D
from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    seed = 3
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

    # start_state = torch.tensor([-0.2275, -0.0472], **tensor_args)
    # goal_state = torch.tensor([0.5302, 0.9507], **tensor_args)

    multi_goal_states = goal_state.unsqueeze(0)

    # Construct cost function
    default_params_env = env.get_chomp_params(robot=robot)
    n_support_points = default_params_env['n_support_points']
    dt = default_params_env['dt']

    cost_collisions = []
    weights_cost_l = []
    for collision_field in task.get_collision_fields():
        cost_collisions.append(
            CostCollision(
                robot, n_support_points,
                field=collision_field,
                sigma_coll=1.0,
                tensor_args=tensor_args
            )
        )
        weights_cost_l.append(10.0)

    cost_func_list = [*cost_collisions]
    cost_composite = CostComposite(
        robot, n_support_points, cost_func_list,
        weights_cost_l=weights_cost_l,
        tensor_args=tensor_args
    )

    num_particles_per_goal = 10
    opt_iters = 100

    planner_params = dict(
        **default_params_env,
        n_dof=robot.q_dim,
        num_particles_per_goal=num_particles_per_goal,
        start_state=start_state,
        multi_goal_states=goal_state.unsqueeze(0),  # add batch dim for interface,
        cost=cost_composite,
        tensor_args=tensor_args,
    )
    planner = CHOMP(**planner_params)

    # Optimize
    trajs_0 = planner.get_traj()
    trajs_iters = torch.empty((opt_iters + 1, *trajs_0.shape), **tensor_args)
    trajs_iters[0] = trajs_0
    with TimerCUDA() as t:
        for i in range(opt_iters):
            trajs = planner.optimize(debug=True)
            trajs_iters[i+1] = trajs
    print(f'Optimization time: {t.elapsed:.3f} sec, per iteration: {t.elapsed/opt_iters:.3f}')

    # -------------------------------- Visualize ---------------------------------
    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_iters[-1])*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_iters[-1])*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_iters[-1])}')

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
