import matplotlib.pyplot as plt
import torch

from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar
from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.utils import create_grid_spheres
from torch_robotics.robot.point_mass_robot import PointMassRobot
from torch_robotics.task.tasks import PlanningTask
from torch_robotics.torch_utils.torch_timer import Timer
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()


if __name__ == "__main__":
    # planner = 'rrt-connect'
    planner = 'rrt-star'

    seed = 0
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    # List of objects composed of primitive shapes
    obj_list = create_grid_spheres(rows=7, cols=7, heights=0, radius=0.1, tensor_args=tensor_args)

    env = EnvBase(
        name='GridCircles2D',
        limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environment limits
        obj_list=obj_list,
        tensor_args=tensor_args
    )

    robot = PointMassRobot(
        q_limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # configuration space limits
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-0.81, -0.81], [0.95, 0.95]], **tensor_args),  # workspace limits
        # use_occupancy_map=True,  # whether to create and evaluate collisions on an occupancy map
        use_occupancy_map=False,
        cell_size=0.01,
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
        n_radius = 0.1
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
        traj, start_state=start_state, goal_state=goal_state, render_planner=True,
        animate=True,
        video_filepath='pointmass_2d_circles.mp4'
    )
    plt.show()
