import os

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar
from torch_robotics.environment.env_spheres_3d import EnvSpheres3D
from torch_robotics.robot.panda_robot import PandaRobot
from torch_robotics.task.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import Timer
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()


if __name__ == "__main__":
    # planner = 'rrt-connect'
    planner = 'rrt-star'

    seed = 2
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
        # use_occupancy_map=True,  # whether to create and evaluate collisions on an occupancy map
        use_occupancy_map=False,
        cell_size=0.05,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    q_free = task.random_coll_free_q(n_samples=2)
    start_state = q_free[0]
    goal_state = q_free[1]

    n_iters = 30000
    step_size = torch.pi/80
    n_radius = torch.pi/4
    max_time = 120

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
        goal_prob = 0.2

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

        planner = RRTStar(**rrt_star_params)
    else:
        raise NotImplementedError

    # Optimize
    with Timer() as t:
        traj = planner.optimize(debug=True, refill_samples_buffer=True)
    print(f'Optimization time: {t.elapsed:.3f} sec')

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        env=env,
        robot=robot,
        planner=planner
    )
    fig, ax = planner_visualizer.render_trajectory(
        traj, start_state=start_state, goal_state=goal_state, render_planner=False,
        animate=True,
        video_filepath=os.path.basename(__file__).replace('.py', '.mp4')
    )
    plt.show()
