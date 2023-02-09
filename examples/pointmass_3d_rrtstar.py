import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np

from experiment_launcher.utils import fix_random_seed
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from mp_baselines.planners.rrt_star import RRTStar


if __name__ == "__main__":
    seed = 7
    fix_random_seed(seed)

    device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float64}

    n_dof = 3
    n_iters = 50000
    max_best_cost_iters = 5000
    step_size = 0.1
    n_radius = 2.
    n_knn = 10
    goal_prob = 0.20
    max_time = 15.
    start_state = torch.tensor([-9, -9, -9], **tensor_args)
    goal_state = torch.tensor([9, 9, 9], **tensor_args)
    limits = torch.tensor([[-10, 10], [-10, 10], [-10, 10]], **tensor_args)

    ## Obstacle map
    # obst_list = [(0, 0, 4, 6)]
    obst_list = []
    cell_size = 0.2
    map_dim = [20, 20, 20]

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        map_type='direct',
        random_gen=True,
        num_obst=15,
        rand_limits=[(-8.5, 8.5), (-8.5, 8.5), (-8.5, 8.5)],
        rand_rect_shape=[4, 4, 4],
        rand_circle_radius=3,
        tensor_args=tensor_args,
    )

    # Obstacle generation
    obst_map, obst_list = generate_obstacle_map(**obst_params)

    # -------------------------------- Planner ---------------------------------
    rrt_params = dict(
        n_dofs=n_dof,
        n_iters=n_iters,
        max_best_cost_iters=max_best_cost_iters,
        start_state=start_state,
        limits=limits,
        cost=obst_map,
        step_size=step_size,
        n_radius=n_radius,
        n_knn=n_knn,
        max_time=max_time,
        goal_prob=goal_prob,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    planner = RRTStar(**rrt_params)

    # ---------------------------------------------------------------------------
    # Optimize
    start = time.time()
    traj = planner.optimize(debug=True, informed=True)
    print(f"{time.time() - start} seconds")

    # ---------------------------------------------------------------------------
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    planner.render(ax)
    obst_map.plot(ax)
    if traj is not None:
        traj = np.array(traj)
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], 'b-')
        ax.scatter3D(traj[:, 0], traj[:, 1], traj[:, 2], color='b')
    ax.scatter3D(start_state[0], start_state[1], start_state[2], 'go', zorder=10, s=100)
    ax.scatter3D(goal_state[0], goal_state[1], goal_state[2], 'ro', zorder=10, s=100)
    ax.set_aspect('equal')
    plt.show()
