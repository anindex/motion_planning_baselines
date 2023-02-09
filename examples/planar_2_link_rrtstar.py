import os

import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from dplan.utils import tensor_linspace
from experiment_launcher.utils import fix_random_seed
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite
from mp_baselines.planners.rrt_star import RRTStar
from torch_planning_objectives.fields.occupancy_map.obst_map import ObstacleCircle


def create_circles():
    radius = 0.4
    n = 3
    circles = np.empty((n, 3), dtype=np.float32)
    circles[:, 2] = radius
    circles[0, :2] = np.array([-0.5, 0.5])
    circles[1, :2] = np.array([0.5, 0.5])
    circles[2, :2] = np.array([0., -0.5])
    obst_list = [ObstacleCircle((x, y), r) for x, y, r in circles]
    return circles, obst_list


def create_circles_random():
    n = 10
    circles = []
    for i in range(n):
        p = np.random.uniform(0.4, 0.5)
        theta = np.random.uniform(np.pi / 8, np.pi - np.pi / 8)
        if np.random.rand() < 0.5:
            theta = -theta
        x = p * np.cos(theta)
        y = p * np.sin(theta)
        r = np.random.uniform(0.1, 0.15)
        circles.append((x, y, r))

    obst_list = [ObstacleCircle((x, y), r) for x, y, r in circles]
    return circles, obst_list



class RobotPlanarTwoLink:

    def __init__(self, tensor_args):
        self.tensor_args = tensor_args
        self.l1 = 0.25
        self.l2 = 0.25
        self.limits = torch.tensor([[-np.pi, np.pi], [-np.pi * 3/4, np.pi * 3/4]], **tensor_args)

    def end_link_positions(self, qs):
        pos_end_link1 = torch.zeros((*qs.shape[0:2], 1, 2), **self.tensor_args)
        pos_end_link2 = torch.zeros((*qs.shape[0:2], 1, 2), **self.tensor_args)

        pos_end_link1[..., 0, 0] = self.l1 * torch.cos(qs[..., 0])
        pos_end_link1[..., 0, 1] = self.l1 * torch.sin(qs[..., 0])

        pos_end_link2[..., 0, 0] = pos_end_link1[..., 0, 0] + self.l2 * torch.cos(qs[..., 0] + qs[..., 1])
        pos_end_link2[..., 0, 1] = pos_end_link1[..., 0, 1] + self.l2 * torch.sin(qs[..., 0] + qs[..., 1])

        return pos_end_link1, pos_end_link2

    def fk_map(self, qs):
        points_along_links = 25
        p1, p2 = self.end_link_positions(qs)
        positions_link1 = tensor_linspace(torch.zeros_like(p1), p1, points_along_links)
        positions_link1 = positions_link1.swapaxes(-3, -1).squeeze(-1)
        positions_link2 = tensor_linspace(p1, p2, points_along_links)
        positions_link2 = positions_link2.swapaxes(-3, -1).squeeze(-1)

        pos_x = torch.cat((positions_link1, positions_link2), dim=-2)
        assert pos_x.ndim == 4, "batch, trajectory, points, x_dim"
        return pos_x

    def render(self, qs, ax):
        p1, p2 = map(np.squeeze, self.end_link_positions(qs))
        ax.plot([0, p1[0]], [0, p1[1]], color='blue', linewidth=1.)
        l2 = torch.vstack((p1, p2))
        ax.plot(l2[:, 0], l2[:, 1], color='blue', linewidth=1.)
        ax.scatter(p2[0], p2[1], color='red', marker='o')


if __name__ == "__main__":
    # seed = 2
    # fix_random_seed(seed)

    # device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float64}

    # -------------------------------- Robot ---------------------------------
    robot = RobotPlanarTwoLink(tensor_args)

    n_dof = 2
    n_iters = 30000
    max_best_cost_iters = 2000
    cost_eps = 1e-2
    step_size = 0.01
    n_radius = 0.1
    n_knn = 5
    goal_prob = 0.1
    max_time = 15.
    limits = robot.limits
    start_state = torch.tensor([0, 0], **tensor_args)
    # start_state = torch.tensor([np.pi/2, 0], **tensor_args)
    goal_state = torch.tensor([np.pi, 0], **tensor_args)

    ## Obstacle map
    cell_size = 0.01
    map_dim = (2, 2)

    # circles, obst_list = create_circles()
    circles, obst_list = create_circles_random()

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        map_type='direct',
        tensor_args=tensor_args,
    )
    obst_map, obst_list = generate_obstacle_map(**obst_params)

    obst_map.plot()
    plt.show()

    # -------------------------------- Planner ---------------------------------
    rrt_params = dict(
        n_dofs=n_dof,
        n_iters=n_iters,
        max_best_cost_iters=max_best_cost_iters,
        cost_eps=cost_eps,
        start_state=start_state,
        limits=limits,
        fk_map=robot.fk_map,
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
    fig, ax = plt.subplots()

    def animate_fn(i):
        ax.clear()

        obst_map.plot(ax)

        q = traj[i]
        q = q.view(1, 1, -1)
        robot.render(q, ax)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')


    animation_time_in_seconds = 5
    print('Creating animation...')
    ani = FuncAnimation(fig, animate_fn,
                        frames=len(traj),
                        interval=animation_time_in_seconds * 1000 / len(traj),
                        repeat=False)
    print('...finished Creating animation')

    print('Saving video...')
    ani.save(os.path.join('planar_2_link_rrtstar.mp4'), fps=int(len(traj) / animation_time_in_seconds), dpi=90)
    print('...finished Saving video')

    plt.show()


