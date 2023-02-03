import math
import sys

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

    def render(self):
        if self.parent is not None:
            x, y = self.config.cpu().numpy(), self.parent.config.cpu().numpy()
            plt.plot([x[0], y[0]], [x[1], y[1]], color='k', linewidth=0.5)
        for child in self.children:
            child.render()

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
            cost=None,
            n_iters_after_success=None,
            step_size: float = 0.1,
            n_radius: float = 1.,
            n_knn: int = 0,
            max_time: float = 60.,
            goal_prob: float = .1,
            goal_state: torch.Tensor = None,
            tensor_args: dict = None
    ):
        super(RRTStar, self).__init__(name='RRTStar', tensor_args=tensor_args)
        self.n_dofs = n_dofs
        self.n_iters = n_iters

        self.n_iters_after_success = n_iters_after_success

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

        self.cost = cost
        self.reset()

    def reset(self):
        self.nodes = []
        self.nodes_torch = None

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
        start_time = time.time()
        iteration = 0
        # best_possible_cost = distance_fn(goal, start) # if straight line is possible
        iters_after_first_success = 0
        while (elapsed_time(start_time) < self.max_time) and (iteration < self.n_iters):
            do_goal = goal_n is None and (iteration == 0 or torch.rand(1) < self.goal_prob)
            s = self.goal_state if do_goal else self.sample_fn(**observation)

            # Informed RRT*
            if informed and (goal_n is not None) and (self.distance_fn(self.start_state, s) + self.distance_fn(s, self.goal_state) >= goal_n.cost):
                continue
            if iteration % print_freq == 0 or iteration % (self.n_iters - 1) == 0:
                success = goal_n is not None
                cost = goal_n.cost if success else torch.inf
                if debug:
                    print('Iteration: {} | Time: {:.3f} | Success: {} | {} | Cost: {:.4f}'.format(
                        iteration, elapsed_time(start_time), success, do_goal, cost))

            iteration += 1

            # nearest node
            # nearest = argmin(lambda n: self.distance_fn(n.config, s), self.nodes)
            distances = self.distance_fn(self.nodes_torch, s)
            min_idx = torch.argmin(distances)
            nearest = self.nodes[min_idx]

            extended = self.extend_fn(nearest.config, s, max_step=self.step_size, max_dist=self.n_radius)
            path = safe_path(extended, self.collision_fn)
            if len(path) == 0:
                continue
            new = OptimalNode(path[-1], parent=nearest, d=self.distance_fn(
                nearest.config, path[-1]), path=list(path[:-1]), iteration=iteration)

            if do_goal and (self.distance_fn(new.config, self.goal_state) < eps):
                goal_n = new
                goal_n.set_solution(True)

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
                nodes_np = np.array(self.nodes)
                neighbors = nodes_np[neighbors_idxs.squeeze()]
                if neighbors_idxs.nelement() == 1:
                    neighbors = [neighbors]
            else:
                neighbors = []

            # TODO: smooth solution once found to improve the cost bound
            # for n in neighbors:
            #     d = self.distance_fn(n.config, new.config)
            #     if (n.cost + d) < new.cost:
            #         extended = self.extend_fn(n.config, new.config, max_step=self.step_size, max_dist=self.n_radius)
            #         n_path = safe_path(extended, self.collision_fn)[:]
            #         if len(n_path) != 0:
            #             n_dist = self.distance_fn(new.config, n_path[-1])
            #             if n_dist < eps:
            #                 new.rewire(n, d, list(n_path[:-1]), iteration=iteration)
            # TODO - avoid repeating work
            for n in neighbors:
                d = self.distance_fn(n.config, new.config)
                if (new.cost + d) < n.cost:
                    extended = self.extend_fn(new.config, n.config, max_step=self.step_size, max_dist=self.n_radius)
                    n_path = safe_path(extended, self.collision_fn)[:]
                    if len(n_path) != 0:
                        n_dist = self.distance_fn(n.config, n_path[-1])
                        if n_dist < eps:
                            n.rewire(new, d, list(n_path[:-1]), iteration=iteration)

            # Stop if some iterations passed after the first success
            success = goal_n is not None
            if success:
                iters_after_first_success += 1

            if self.n_iters_after_success is not None:
                if iters_after_first_success > self.n_iters_after_success:
                    break

        if goal_n is None:
            return None
        path = goal_n.retrace()
        # return torch.stack(path, dim=0)
        return purge_duplicates_from_traj(path, eps=eps)

    def check_point_collision(self, pos, **observation):
        if pos.ndim == 1:
            pos = pos.unsqueeze(0).unsqueeze(0)  # add batch and traj len dim for interface
        elif pos.ndim == 2:
            pos = pos.unsqueeze(1)  # add traj len dim for interface
        # check in bounds
        if torch.any(torch.logical_or(torch.less(pos[0, 0, :], self.limits[:, 0]),
                                      torch.greater(pos[0, 0, :], self.limits[:, 1]))):
            return True
        # do forward kinematics
        fk_map = observation.get('fk', None)
        pos_x = None
        if fk_map is not None:
            pos_x = fk_map(pos)

        # collision in task space
        collision = self.cost.get_collisions(pos, x_trajs=pos_x, **observation).squeeze()
        return collision > 0

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
        # sample a batch of random points and pick one that is not in collision
        reject = True
        for i in range(1000):
            pos = torch.rand((100, self.n_dofs), **self.tensor_args)
            pos = self.limits[:, 0] + pos * (self.limits[:, 1] - self.limits[:, 0])
            in_collision = self.check_point_collision(pos, **observation)
            idxs_not_in_collision = torch.argwhere(in_collision == False)
            if idxs_not_in_collision.nelement() == 0:
                # all points are in collision
                continue
            # pick a random point not in collision
            idx_random = torch.randperm(len(idxs_not_in_collision))[0].item()
            pos = pos[idxs_not_in_collision[idx_random]].squeeze()
            reject = False
            break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return pos

    def collision_fn(self, pos, pos_x=None, **observation):
        return self.check_point_collision(pos, pos_x=pos_x, **observation)

    def sample_fn(self, check_collision=True, **observation):
        if check_collision:
            return self.random_collision_free(**observation)
        else:
            pos = torch.rand(self.n_dofs, **self.tensor_args)
            pos = self.limits[:, 0] + pos * (self.limits[:, 1] - self.limits[:, 0])
            return pos

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

    def render(self):
        self.nodes[0].render()

    def _get_costs(self, state_trajectories, **observation):
        if self.cost is None:
            costs = torch.zeros(self.num_samples, )
        else:
            costs = self.cost.eval(state_trajectories, **observation)
        return costs


