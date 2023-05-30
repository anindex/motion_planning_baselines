
from abc import ABC, abstractmethod

import einops
import torch

from mp_baselines.planners.costs.factors.field_factor import FieldFactor
from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor


class Cost(ABC):
    def __init__(self, robot, traj_len, tensor_args=None, **kwargs):
        self.robot = robot
        self.n_dof = robot.q_dim
        self.dim = 2 * self.n_dof  # position + velocity
        self.traj_len = traj_len

        self.tensor_args = tensor_args

    def set_cost_factors(self):
        pass

    def __call__(self, trajs, **kwargs):
        return self.eval(trajs, **kwargs)

    @abstractmethod
    def eval(self, trajs, **kwargs):
        pass

    @abstractmethod
    def get_linear_system(self, trajs, **kwargs):
        pass

    def get_q_pos_vel_and_fk_map(self, trajs, **kwargs):
        assert trajs.ndim == 3 or trajs.ndim == 4
        N = 1
        if trajs.ndim == 4:
            N, B, H, D = trajs.shape  # n_goals (or steps), batch of trajectories, length, dim
            trajs = einops.rearrange(trajs, 'N B H D -> (N B) H D')
        else:
            B, H, D = trajs.shape

        q_pos = self.robot.get_position(trajs)
        q_vel = self.robot.get_velocity(trajs)
        H = self.robot.fk_map(q_pos)  # I, taskspaces, x_dim+1, x_dim+1 (homogeneous transformation matrices)
        return trajs, q_pos, q_vel, H


class CostComposite(Cost):

    def __init__(
        self,
        robot,
        traj_len,
        cost_list,
        **kwargs
    ):
        super().__init__(robot, traj_len, **kwargs)
        self.cost_list = cost_list

    def eval(self, trajs, **kwargs):
        trajs, q_pos, q_vel, H = self.get_q_pos_vel_and_fk_map(trajs)

        costs = 0
        for cost in self.cost_list:
            costs += cost(trajs, q_pos=q_pos, q_vel=q_vel, H=H, **kwargs)

        return costs

    def get_linear_system(self, trajs, **kwargs):
        trajs.requires_grad = True
        trajs, q_pos, q_vel, H = self.get_q_pos_vel_and_fk_map(trajs)

        batch_size = trajs.shape[0]
        As, bs, Ks = [], [], []
        optim_dim = 0
        for cost in self.cost_list:
            A, b, K = cost.get_linear_system(trajs, H=H, **kwargs)
            if A is None or b is None or K is None:
                continue
            optim_dim += A.shape[1]
            As.append(A.detach())
            bs.append(b.detach())
            Ks.append(K.detach())
        A = torch.cat(As, dim=1)
        b = torch.cat(bs, dim=1)
        K = torch.zeros(batch_size, optim_dim, optim_dim, **self.tensor_args)
        offset = 0
        for i in range(len(Ks)):
            dim = Ks[i].shape[1]
            K[:, offset:offset+dim, offset:offset+dim] = Ks[i]
            offset += dim
        return A, b, K


class CostCollision(Cost):

    def __init__(
            self,
            robot,
            traj_len,
            field=None,
            sigma_coll=None,
            **kwargs
    ):
        super().__init__(robot, traj_len, **kwargs)
        self.field = field
        self.sigma_coll = sigma_coll

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.obst_factor = FieldFactor(
            self.n_dof,
            self.sigma_coll,
            [1, self.traj_len]
        )

    def eval(self, trajs, q_pos=None, q_vel=None, H=None, **observation):
        costs = 0
        if self.field is not None:
            H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            err_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                H_pos=H_pos,
                calc_jacobian=False,  # TODO: NOTE(an): no need for grads in StochGPMP
                obstacle_spheres=observation.get('obstacle_spheres', None)
            )
            w_mat = self.obst_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs

    def get_linear_system(self, trajs, H=None, **observation):
        A, b, K = None, None, None
        if self.field is not None:
            batch_size = trajs.shape[0]
            H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            A = torch.zeros(batch_size, (self.traj_len - 1), self.dim * self.traj_len, **self.tensor_args)
            err_obst, H_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                H_pos=H_pos,
                calc_jacobian=True,
                obstacle_spheres=observation.get('obstacle_spheres', None)
            )
            for i in range(self.traj_len - 1):
                A[:, i, (i + 1) * self.dim:(i + 1) * self.dim + self.n_dof] = H_obst[:, i]
            b = err_obst.unsqueeze(-1)
            K = self.obst_factor.K * torch.eye((self.traj_len - 1), **self.tensor_args).repeat(batch_size, 1, 1)
        return A, b, K


class CostGP(Cost):

    def __init__(
        self,
        robot,
        traj_len,
        start_state,
        dt,
        sigma_params,
        **kwargs
    ):
        super().__init__(robot, traj_len, **kwargs)
        self.start_state = start_state
        self.dt = dt

        self.sigma_start = sigma_params['sigma_start']
        self.sigma_gp = sigma_params['sigma_gp']

        self.set_cost_factors()

    def set_cost_factors(self):

        #========= Cost factors ===============
        self.start_prior = UnaryFactor(
            self.dim,
            self.sigma_start,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

    def eval(self, trajs, **observation):
        # trajs = trajs.reshape(-1, self.traj_len, self.dim)
        # Start cost
        err_p = self.start_prior.get_error(trajs[:, [0]], calc_jacobian=False)
        w_mat = self.start_prior.K
        start_costs = err_p @ w_mat.unsqueeze(0) @ err_p.transpose(1, 2)
        start_costs = start_costs.squeeze()

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()

        costs = start_costs + gp_costs

        return costs
    
    def get_linear_system(self, trajs, **observation):
        batch_size = trajs.shape[0]
        A = torch.zeros(batch_size, self.dim * self.traj_len, self.dim * self.traj_len, **self.tensor_args)
        b = torch.zeros(batch_size, self.dim * self.traj_len, 1, **self.tensor_args)
        K = torch.zeros(batch_size, self.dim * self.traj_len, self.dim * self.traj_len, **self.tensor_args)

        # Start prior factor
        err_p, H_p = self.start_prior.get_error(trajs[:, [0]])
        A[:, :self.dim, :self.dim] = H_p
        b[:, :self.dim] = err_p
        K[:, :self.dim, :self.dim] = self.start_prior.K

        # GP factors
        err_gp, H1_gp, H2_gp = self.gp_prior.get_error(trajs)
        for i in range(self.traj_len - 1):
            A[:, (i+1)*self.dim:(i+2)*self.dim, i*self.dim:(i+1)*self.dim] = H1_gp[[i]]
            A[:, (i+1)*self.dim:(i+2)*self.dim, (i+1)*self.dim:(i+2)*self.dim] = H2_gp[[i]]
            b[:, (i+1)*self.dim:(i+2)*self.dim] = err_gp[:, i]
            K[:, (i+1)*self.dim:(i+2)*self.dim, (i+1)*self.dim:(i+2)*self.dim] = self.gp_prior.Q_inv[[i]]

        return A, b, K


class CostGPTrajectory(Cost):

    def __init__(
            self,
            robot,
            traj_len,
            start_state,
            dt,
            sigma_params,
            **kwargs
    ):
        super().__init__(robot, traj_len, **kwargs)
        self.start_state = start_state
        self.dt = dt

        self.sigma_gp = sigma_params['sigma_gp']

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

    def eval(self, trajs, **observation):
        # trajs = trajs.reshape(-1, self.traj_len, self.dim)

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()

        costs = gp_costs

        return costs

    def get_linear_system(self, trajs, **observation):
        pass


class CostGoal(Cost):

    def __init__(
        self,
        robot,
        traj_len,
        field=None,
        sigma_goal=None,
        **kwargs
    ):
        super().__init__(robot, traj_len, **kwargs)
        self.field = field
        self.sigma_goal = sigma_goal

        self.set_cost_factors()

    def set_cost_factors(self):

        #========= Cost factors ===============
        self.goal_factor = FieldFactor(
            self.n_dof,
            self.sigma_goal,
            [self.traj_len - 1, self.traj_len]   # only take last point
        )

    def eval(self, trajs, x_trajs=None, **observation):
        costs = 0
        if self.field is not None:
            err_obst = self.goal_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=False,  # NOTE(an): no need for grads in StochGPMP
            )
            w_mat = self.goal_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs

    def get_linear_system(self, trajs, x_trajs=None, **observation):
        A, b, K = None, None, None
        if self.field is not None:
            batch_size = trajs.shape[0]
            A = torch.zeros(batch_size, 1, self.dim * self.traj_len, **self.tensor_args)
            err_goal, H_goal = self.goal_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=True,
            )
            A[:, :, -self.dim:(-self.dim + self.n_dof)] = H_goal
            b = err_goal.unsqueeze(-1)
            K = self.goal_factor.K * torch.eye(1, **self.tensor_args).repeat(batch_size, 1, 1)
        return A, b, K


class CostGoalPrior(Cost):

    def __init__(
        self,
        robot,
        traj_len,
        multi_goal_states=None,  # num_goal x n_dim (pos + vel)
        num_particles_per_goal=None,
        num_samples=None,
        sigma_goal_prior=None,
        **kwargs
    ):
        super().__init__(robot, traj_len, **kwargs)
        self.multi_goal_states = multi_goal_states
        self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.num_samples = num_samples
        self.sigma_goal_prior = sigma_goal_prior

        self.set_cost_factors()

    def set_cost_factors(self):

        self.multi_goal_prior = []
        for i in range(self.num_goals):
            self.multi_goal_prior.append(
                UnaryFactor(
                    self.dim,
                    self.sigma_goal_prior,
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

    def eval(self, trajs, **observation):
        costs = 0
        if self.multi_goal_states is not None:
            x = trajs.reshape(self.num_goals, self.num_particles_per_goal * self.num_samples, self.traj_len, self.dim)
            costs = torch.zeros(self.num_goals, self.num_particles_per_goal * self.num_samples, **self.tensor_args)
            for i in range(self.num_goals):
                err_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]], calc_jacobian=False)
                w_mat = self.multi_goal_prior[i].K
                goal_costs = err_g @ w_mat.unsqueeze(0) @ err_g.transpose(1, 2)
                goal_costs = goal_costs.squeeze()
                costs[i] += goal_costs
            costs = costs.flatten()
        return costs

    def get_linear_system(self, trajs, **observation):
        A, b, K = None, None, None
        if self.multi_goal_states is not None:
            npg = self.num_particles_per_goal
            batch_size = npg * self.num_goals
            x = trajs.reshape(self.num_goals, self.num_particles_per_goal, self.traj_len, self.dim)
            A = torch.zeros(batch_size, self.dim, self.dim * self.traj_len, **self.tensor_args)
            b = torch.zeros(batch_size, self.dim, 1, **self.tensor_args)
            K = torch.zeros(batch_size, self.dim, self.dim, **self.tensor_args)
            # TODO: remove this for loop
            for i in range(self.num_goals):
                err_g, H_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]])
                A[i*npg: (i+1)*npg, :, -self.dim:] = H_g
                b[i*npg: (i+1)*npg] = err_g
                K[i*npg: (i+1)*npg] = self.multi_goal_prior[i].K
        return A, b, K