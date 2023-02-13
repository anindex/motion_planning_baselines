import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np

from experiment_launcher.utils import fix_random_seed
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from mp_baselines.planners.rrt_star import RRTStar
from torch_planning_objectives.fields.occupancy_map.obst_map import ObstacleCircle, ObstacleRectangle


def fixed_rectangles():
    rectangles_bl_tr = list()
    # bottom left xy, top right xy coordinates
    halfside = 0.2
    rectangles_bl_tr.append((-0.75, -1.0, -0.75 + halfside, -0.3))
    rectangles_bl_tr.append((-0.75, 0.3, -0.75 + halfside, 1.0))
    rectangles_bl_tr.append((-0.75, -0.15, -0.75 + halfside, 0.15))

    rectangles_bl_tr.append((-halfside/2 - 0.2, -1, halfside/2 - 0.2, -1+0.05))
    rectangles_bl_tr.append((-halfside/2 - 0.2, -1+0.2, halfside/2 - 0.2, 1-0.2))
    rectangles_bl_tr.append((-halfside/2 - 0.2, 1-0.05, halfside/2 - 0.2, 1))

    rectangles_bl_tr.append((-halfside / 2 + 0.2, -1.0, halfside / 2 + 0.2, -0.3))
    rectangles_bl_tr.append((-halfside / 2 + 0.2, 0.3, halfside / 2 + 0.2, 1.0))
    rectangles_bl_tr.append((-halfside / 2 + 0.2, -0.15, halfside / 2 + 0.2, 0.15))

    rectangles_bl_tr.append((0.75-halfside, -1, 0.75, -0.6))
    rectangles_bl_tr.append((0.75-halfside, -0.5, 0.75, -0.2))
    rectangles_bl_tr.append((0.75-halfside, -0.1, 0.75, 0.1))
    rectangles_bl_tr.append((0.75-halfside, 0.2, 0.75, 0.5))
    rectangles_bl_tr.append((0.75-halfside, 0.6, 0.75, 1))

    obst_list = []
    rectangles = []
    for rectangle in rectangles_bl_tr:
        bl_x, bl_y, tr_x, tr_y = rectangle
        x = bl_x + abs(tr_x - bl_x) / 2
        y = bl_y + abs(tr_y - bl_y) / 2
        z = 0
        w = abs(tr_x - bl_x)
        h = abs(tr_y - bl_y)
        d = 1.95
        rect = ((x, y, z), (w, h, d))
        obst_list.append(ObstacleRectangle(*rect))
        rectangles.append(rect)

    return obst_list



if __name__ == "__main__":
    seed = 18
    fix_random_seed(seed)

    # device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float64}

    n_dof = 3
    n_iters = 50000
    max_best_cost_iters = 5000
    step_size = 0.05
    n_radius = 0.1
    n_knn = 5
    goal_prob = 0.1
    max_time = 60.

    start_state = torch.tensor([-0.9, -0.9, -0.9], **tensor_args)
    goal_state = torch.tensor([0.9, 0.9, 0.9], **tensor_args)
    limits = torch.tensor([[-1, 1], [-1, 1], [-1, 1]], **tensor_args)

    ## Obstacle map
    cell_size = 0.05
    map_dim = [2, 2, 2]

    obst_list = fixed_rectangles()

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        map_type='direct',
        tensor_args=tensor_args,
    )
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
    # planner.render(ax)
    obst_map.plot(ax)
    if traj is not None:
        traj = np.array(traj)
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], 'b-')
        ax.scatter3D(traj[:, 0], traj[:, 1], traj[:, 2], color='b')
    ax.scatter3D(start_state[0], start_state[1], start_state[2], 'go', zorder=10, s=100)
    ax.scatter3D(goal_state[0], goal_state[1], goal_state[2], 'ro', zorder=10, s=100)

    # ax.view_init(azim=0, elev=90)
    ax.set_aspect('equal')
    plt.show()


