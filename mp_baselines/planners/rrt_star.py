import math
import sys
from copy import copy
from operator import itemgetter

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from mp_baselines.planners.base import MPPlanner
from mp_baselines.planners.utils import elapsed_time, argmin, safe_path, purge_duplicates_from_traj


class OptimalNode:

    def __init__(self, config, parent=None, d=0, path=[], iteration=None):
        self.config = config
        self.parent = parent
        self.children = set()
        self.d = d
        self.path = path
        if parent is not None:
            self.cost = parent.cost + d
            self.parent.children.add(self)
        else:
            self.cost = d
        self.solution = False
        self.creation = iteration
        self.last_rewire = iteration

    def set_solution(self, solution):
        if self.solution is solution:
            return
        self.solution = solution
        if self.parent is not None:
            self.parent.set_solution(solution)

    def retrace(self):
        if self.parent is None:
            return self.path + [self.config]
        return self.parent.retrace() + self.path + [self.config]

    def rewire(self, parent, d, path, iteration=None):
        if self.solution:
            self.parent.set_solution(False)
        self.parent.children.remove(self)
        self.parent = parent
        self.parent.children.add(self)
        if self.solution:
            self.parent.set_solution(True)
        self.d = d
        self.path = path
        self.update()
        self.last_rewire = iteration

    def update(self):
        self.cost = self.parent.cost + self.d
        for n in self.children:
            n.update()

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def render(self, ax):
        assert ax is not None, "Axis cannot be None"
        if self.parent is not None:
            if ax.name == '3d':
                x, y = self.config.cpu().numpy(), self.parent.config.cpu().numpy()
                ax.plot3D([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], color='k', linewidth=0.5)
            else:
                x, y = self.config.cpu().numpy(), self.parent.config.cpu().numpy()
                ax.plot([x[0], y[0]], [x[1], y[1]], color='k', linewidth=0.5)
        for child in self.children:
            child.render(ax)

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.config) + ')'
    __repr__ = __str__


class RRTStar(MPPlanner):

    def __init__(
            self,
            n_dofs: int,
            n_iters: int,
            start_state: torch.Tensor,
            limits: torch.Tensor,
            fk_map=None,
            cost=None,
            n_iters_after_success=None,
            max_best_cost_iters: int = 1000,
            cost_eps: float = 1e-2,
            step_size: float = 0.1,
            n_radius: float = 1.,
            n_knn: int = 0,
            max_time: float = 60.,
            goal_prob: float = .1,
            goal_state: torch.Tensor = None,
            tensor_args: dict = None,
            n_pre_samples=1000,
            pre_samples=None
    ):
        super(RRTStar, self).__init__(name='RRTStar', tensor_args=tensor_args)
        self.n_dofs = n_dofs
        self.n_iters = n_iters

        self.fk_map = fk_map

        self.n_iters_after_success = n_iters_after_success
        self.max_best_cost_iters = max_best_cost_iters if max_best_cost_iters is not None else n_iters
        self.cost_eps = cost_eps

        # RRTStar params
        self.step_size = step_size
        self.n_radius = n_radius
        assert n_knn >= 0, "knn parameter is < 0"
        self.n_knn = n_knn
        self.max_time = max_time
        self.goal_prob = goal_prob

        self.start_state = start_state
        self.goal_state = goal_state
        self.limits = limits  # [min, max] for each dimension

        self.n_pre_samples = n_pre_samples
        self.pre_samples = pre_samples
        self.last_sample_idx = None

        self.cost = cost
        self.reset()

    def reset(self):
        self.nodes = []
        self.nodes_torch = None

        # Create pre collision-free samples
        n_uniform_samples = self.n_pre_samples - (self.pre_samples.shape[0] if self.pre_samples is not None else 0)
        uniform_samples = self.create_uniform_samples(n_uniform_samples)
        if self.pre_samples is not None:
            self.pre_samples = torch.cat((self.pre_samples, uniform_samples), dim=0)
        else:
            self.pre_samples = uniform_samples

    def create_uniform_samples(self, n_samples, max_samples=1000, **observation):
        max_tries = 1000
        reject = True
        samples = torch.zeros((n_samples, self.n_dofs), **self.tensor_args)
        idx_begin = 0
        for i in range(max_tries):
            qs = torch.rand((max_samples, self.n_dofs), **self.tensor_args)
            qs = self.limits[:, 0] + qs * (self.limits[:, 1] - self.limits[:, 0])
            in_collision = self.check_point_collision(qs, **observation)
            idxs_not_in_collision = torch.argwhere(in_collision == False)
            if idxs_not_in_collision.nelement() == 0:
                # all points are in collision
                continue
            idx_random = torch.randperm(len(idxs_not_in_collision))[:n_samples]
            free_qs = qs[idxs_not_in_collision[idx_random]].squeeze()
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[:idx_end - idx_begin]
            idx_begin = idx_end
            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze()

    def remove_last_sample(self):
        # https://discuss.pytorch.org/t/how-to-remove-an-element-from-a-1-d-tensor-by-index/23109/3
        if len(self.pre_samples) > 0:
            i = self.last_sample_idx
            self.pre_samples = torch.cat([self.pre_samples[:i], self.pre_samples[i+1:]])

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):
        """
        Optimize for best trajectory at current state
        """
        return self._run_optimization(opt_iters, **observation)

    def _run_optimization(self, opt_iters, **observation):
        """
        Run optimization iterations.
        """
        optim_vis = observation.get('optim_vis', False)
        initial_nodes = observation.get('initial_nodes', None)
        informed = observation.get('informed', False)
        eps = observation.get('eps', 1e-6)
        print_freq = observation.get('print_freq', 150)
        debug = observation.get('debug', False)
        if self.collision_fn(self.start_state) or self.collision_fn(self.goal_state):
            return None
        self.nodes = initial_nodes if initial_nodes is not None else [OptimalNode(self.start_state)]
        self.nodes_torch = initial_nodes if initial_nodes is not None else OptimalNode(self.start_state).config
        goal_n = None
        iteration = -1
        iters_after_first_success = 0
        best_cost_iters = 0
        best_cost_eps = torch.inf
        success = False

        # best_possible_cost = distance_fn(goal, start) # if straight line is possible
        start_time = time.time()
        while (elapsed_time(start_time) < self.max_time) and (iteration < self.n_iters):
            iteration += 1

            # Stop if the cost does not improve over iterations
            if best_cost_iters >= self.max_best_cost_iters:
                break
            if goal_n is not None:
                if goal_n.cost < best_cost_eps - self.cost_eps:
                    best_cost_eps = copy(goal_n.cost)
                    best_cost_iters = 0
                else:
                    best_cost_iters += 1

            # Stop if the cost does not improve over  iterations passed after the first success
            success = goal_n is not None
            if success:
                iters_after_first_success += 1
            if self.n_iters_after_success is not None:
                if iters_after_first_success > self.n_iters_after_success:
                    break

            # Sample new node
            do_goal = goal_n is None and (iteration == 0 or torch.rand(1) < self.goal_prob)
            if do_goal:
                s = self.goal_state
            else:
                s = self.sample_fn(**observation)

            if iteration % print_freq == 0 or iteration % (self.n_iters - 1) == 0 or iters_after_first_success == 1:
                if debug:
                    self.print_info(iteration, start_time, success, goal_n)

            # Informed RRT*
            if informed and (goal_n is not None) and (self.distance_fn(self.start_state, s) + self.distance_fn(s, self.goal_state) >= goal_n.cost):
                self.remove_last_sample()
                continue

            # nearest node to the sampled node
            # nearest = argmin(lambda n: self.distance_fn(n.config, s), self.nodes)
            distances = self.distance_fn(self.nodes_torch, s)
            min_idx = torch.argmin(distances)
            nearest = self.nodes[min_idx]

            # create a safe path from the sampled node to the nearest node
            extended = self.extend_fn(nearest.config, s, max_step=self.step_size, max_dist=self.n_radius)
            path = safe_path(extended, self.collision_fn)
            if len(path) == 0:
                continue

            if not do_goal and torch.allclose(path[-1], s):
                self.remove_last_sample()

            new = OptimalNode(path[-1], parent=nearest, d=self.distance_fn(
                nearest.config, path[-1]), path=list(path[:-1]), iteration=iteration)

            if do_goal and (self.distance_fn(new.config, self.goal_state) < eps):
                goal_n = new
                goal_n.set_solution(True)

            # Append the node to the tree
            self.nodes.append(new)
            self.nodes_torch = torch.vstack((self.nodes_torch, new.config))

            # Get neighbors of new node
            distances = self.distance_fn(self.nodes_torch, new.config)
            if self.n_knn > 0:
                # https://discuss.pytorch.org/t/k-nearest-neighbor-in-pytorch/59695
                knn = distances.topk(min(self.n_knn, len(distances)), largest=False)
                neighbors_idxs = knn.indices
            else:
                neighbors_idxs = torch.argwhere(distances < self.n_radius)

            if neighbors_idxs.nelement() != 0:
                neighbors_idxs = np.atleast_1d(neighbors_idxs.squeeze().cpu().numpy())
                try:
                    neighbors = list(itemgetter(*neighbors_idxs)(self.nodes))
                except TypeError:
                    neighbors = [itemgetter(*neighbors_idxs)(self.nodes)]
            else:
                neighbors = []

            # Rewire - update parents
            for n in neighbors:
                d = self.distance_fn(n.config, new.config)
                if (new.cost + d) < n.cost:
                    extended = self.extend_fn(new.config, n.config, max_step=self.step_size, max_dist=self.n_radius)
                    n_path = safe_path(extended, self.collision_fn)
                    if len(n_path) != 0:
                        n_dist = self.distance_fn(n.config, n_path[-1])
                        if n_dist < eps:
                            n.rewire(new, d, list(n_path[:-1]), iteration=iteration)

        self.print_info(iteration, start_time, success, goal_n)

        if goal_n is None:
            return None

        # get path from start to goal
        path = goal_n.retrace()

        return purge_duplicates_from_traj(path, eps=1e-6)

    def print_info(self, iteration, start_time, success, goal_n):
        print(f'Iteration: {iteration} | Nodes: {self.nodes_torch.shape[0]} | Time: {elapsed_time(start_time):.3f} '
              f'| Success: {success} | Cost: {goal_n.cost if success else torch.inf:.6f}')

    def check_point_collision(self, qs, **observation):
        if qs.ndim == 1:
            qs = qs.unsqueeze(0).unsqueeze(0)  # add batch and trajectory_length dimension for interface
        elif qs.ndim == 2:
            qs = qs.unsqueeze(1)  # add trajectory_length dimension for interface

        collisions = torch.ones(qs.shape[0], **self.tensor_args)

        # check in bounds
        idxs_in_bounds = torch.argwhere(torch.all(
            torch.logical_and(torch.greater_equal(qs[:, 0, :], self.limits[:, 0]),
                              torch.less_equal(qs[:, 0, :], self.limits[:, 1])
                              ), dim=-1)
        )
        idxs_in_bounds = idxs_in_bounds.squeeze()
        collisions[idxs_in_bounds] = 0

        # check if all points are out of bounds (in collision)
        if torch.count_nonzero(collisions) == qs.shape[0]:
            return collisions

        # do forward kinematics
        if idxs_in_bounds.ndim == 0:
            qs_try = qs[idxs_in_bounds][None, ...]
        else:
            qs_try = qs[idxs_in_bounds]

        # if there is no forward kinematics, then assume it's the identity
        pos_x = qs_try
        if self.fk_map is not None:
            pos_x = self.fk_map(qs_try)

        # collision in task space
        collisions_task_space = torch.zeros_like(collisions[idxs_in_bounds])
        collisions_pos_x = self.cost.get_collisions(pos_x, **observation).squeeze()
        if collisions_pos_x.ndim == 2:
            # configuration is not valid if any point in the task space is in collision
            idx_collisions = torch.argwhere(torch.any(collisions_pos_x, dim=-1)).squeeze()
        else:
            idx_collisions = torch.argwhere(collisions_pos_x)

        if idx_collisions.nelement() > 0:
            collisions_task_space[idx_collisions] = 1

        # filter collisions in task space
        collisions = torch.logical_or(collisions[idxs_in_bounds], collisions_task_space)

        return collisions

    def check_line_collision(self, q1, q2, num_interp_points=15, **observation):
        alpha = torch.linspace(0, 1, num_interp_points + 2)[1:-1].to(**self.tensor_args)  # skip first and last
        q1 = q1.unsqueeze(0)
        q2 = q2.unsqueeze(0)
        points = q1 + (q2 - q1) * alpha.unsqueeze(1)
        return (self.check_point_collision(points, **observation) > 0.).any()
    
    def random_collision_free(self, **observation):
        """
        Returns: random positions in environment space not in collision
        """
        if len(self.pre_samples) > 0:
            idx = torch.randperm(len(self.pre_samples))[0]
            qs = self.pre_samples[idx]
            self.last_sample_idx = idx
        else:
            qs = self.create_uniform_samples(1, **observation)

        return qs

    def collision_fn(self, qs, pos_x=None, **observation):
        return self.check_point_collision(qs, pos_x=pos_x, **observation)

    def sample_fn(self, check_collision=True, **observation):
        if check_collision:
            return self.random_collision_free(**observation)
        else:
            qs = torch.rand(self.n_dofs, **self.tensor_args)
            qs = self.limits[:, 0] + qs * (self.limits[:, 1] - self.limits[:, 0])
            return qs

    def distance_fn(self, q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    def extend_fn(self, q1, q2, max_step=0.03, max_dist=0.1):
        # max_dist must be <= radius of RRT star!
        dist = self.distance_fn(q1, q2)
        if dist > max_dist:
            q2 = q1 + (q2 - q1) * (max_dist / dist)

        alpha = torch.linspace(0, 1, int(dist / max_step) + 2).to(**self.tensor_args)  # skip first and last
        q1 = q1.unsqueeze(0)
        q2 = q2.unsqueeze(0)
        extension = q1 + (q2 - q1) * alpha.unsqueeze(1)
        return extension

    def render(self, ax):
        self.nodes[0].render(ax)

    def _get_costs(self, state_trajectories, **observation):
        if self.cost is None:
            costs = torch.zeros(self.num_samples, )
        else:
            costs = self.cost.eval(state_trajectories, **observation)
        return costs



