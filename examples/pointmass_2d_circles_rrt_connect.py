import time

import matplotlib.pyplot as plt
import torch

from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.utils import elapsed_time
from torch_robotics.environment.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import to_numpy, get_torch_device

if __name__ == "__main__":
    seed = 5
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # -------------------------------- Environment ---------------------------------
    ws_limits = torch.tensor([[-1, 1], [-1, 1]], **tensor_args)

    # Obstacles
    cell_size = 0.01
    map_dim = [2, 2]

    rows = 7
    cols = 7
    radius = 0.075
    obj_list = create_grid_spheres(rows=rows, cols=cols, heights=0, radius=radius, tensor_args=tensor_args)

    env_params = dict(
        map_dim=map_dim,
        obj_list=obj_list,
        cell_size=cell_size,
        map_type='direct',
        tensor_args=tensor_args,
    )
    obst_map = build_obstacle_map(**env_params)

    env = ObstacleMapEnv(
        name='GridCircles',
        q_n_dofs=2,
        q_min=ws_limits[:, 0],
        q_max=ws_limits[:, 1],
        obstacle_map=obst_map,
        tensor_args=tensor_args
    )

    robot = PointMassRobot()

    task = Task(env=env, robot=robot)


    # -------------------------------- Planner ---------------------------------
    n_iters = 30000
    step_size = 0.01
    n_radius = 0.1
    max_time = 60.

    start_state = torch.tensor([-0.8, -0.8], **tensor_args)
    goal_state = torch.tensor([0.8, 0.8], **tensor_args)

    rrt_params = dict(
        env=env,
        n_iters=n_iters,
        start_state=start_state,
        step_size=step_size,
        n_radius=n_radius,
        max_time=max_time,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    planner = RRTConnect(**rrt_params)

    # Optimize
    start = time.time()
    traj = planner.optimize(debug=True, informed=True, refill_samples_buffer=True)
    print(f"{elapsed_time(start)} seconds")

    # -------------------------------- Plotting ---------------------------------
    fig, ax = plt.subplots()
    planner.render(ax)
    obst_map.plot(ax)
    if traj is not None:
        traj = to_numpy(traj)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', markersize=3)
    ax.plot(to_numpy(start_state[0]), to_numpy(start_state[1]), 'go', markersize=7)
    ax.plot(to_numpy(goal_state[0]), to_numpy(goal_state[1]), 'ro', markersize=7)
    ax.set_aspect('equal')
    plt.show()


