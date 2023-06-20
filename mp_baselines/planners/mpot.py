import torch
from copy import copy

import einops
import numpy as np

from mp_baselines.planners.base import OptimizationPlanner
from mp_baselines.planners.ot.optimizer import optimize
from mp_baselines.planners.utils.polytopes import POLYTOPE_MAP
from mp_baselines.planners.utils.misc import MinMaxCenterScaler
from mp_baselines.planners.costs.factors.mp_priors_multi import MultiMPPrior
from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor
from mp_baselines.planners.costs.cost_functions import CostGPMPOT, CostGoalPrior, CostCollision, CostComposite

try:
    import cholespy
except ModuleNotFoundError:
    pass

from torch_robotics.torch_utils.torch_timer import Timer


def build_mpot_cost_composite(
    robot=None,
    traj_len=None,
    dt=None,
    start_state=None,
    multi_goal_states=None,
    num_particles_per_goal=None,
    collision_fields=None,
    extra_costs=[],
    sigma_gp=1e-2,
    sigma_coll=1e-5,
    sigma_goal_prior=1e-5,
    w_smooth=1e-7,
    probe_range=[0, 1],
    tensor_args=None,
    **kwargs,
):
    """
    Construct cost composite function for MPOT
    """
    cost_func_list = []

    # GP cost
    start_state_zero_vel = torch.cat((start_state, torch.zeros(start_state.nelement(), **tensor_args)))
    cost_gp_prior = CostGPMPOT(
        robot, traj_len, start_state_zero_vel, dt,
        sigma_gp,
        probe_range,
        weight=w_smooth,
        tensor_args=tensor_args
    )
    cost_func_list.append(cost_gp_prior)

    # Goal state cost
    # if multi_goal_states is not None:
    #     multi_goal_states_zero_vel = torch.cat((multi_goal_states, torch.zeros_like(multi_goal_states)),
    #                                             dim=-1).unsqueeze(0)  # add batch dim for interface
    #     cost_goal_prior = CostGoalPrior(
    #         robot, traj_len, multi_goal_states=multi_goal_states_zero_vel,
    #         num_particles_per_goal=num_particles_per_goal,
    #         sigma_goal_prior=sigma_goal_prior,
    #         tensor_args=tensor_args
    #     )
    #     cost_func_list.append(cost_goal_prior)

    # Collision costs
    for collision_field in collision_fields:
        cost_collision = CostCollision(
            robot, traj_len,
            field=collision_field,
            sigma_coll=sigma_coll,
            tensor_args=tensor_args
        )
        cost_func_list.append(cost_collision)

    # Other costs
    if extra_costs:
        cost_func_list.append(*extra_costs)

    cost_composite = CostComposite(
        robot, traj_len, cost_func_list,
        tensor_args=tensor_args
    )
    return cost_composite
        



class MPOT(OptimizationPlanner):

    def __init__(
            self,
            robot=None,
            n_dof: int = None,
            traj_len: int = None,
            num_particles_per_goal: int = None,
            opt_iters: int = None,
            dt=0.02,
            start_state=None,
            step_size=0.1,
            multi_goal_states=None,
            initial_particle_means=None,
            sigma_start_init=None,
            sigma_goal_init=None,
            sigma_gp_init=None,
            solver_params=None,
            probe_radius=0.2,
            num_bpoint=100,
            pos_limits=[-10, 10],
            vel_limits=[-10, 10],
            random_init=True,
            polytope='orthoplex',
            random_step=False,
            annealing=False,
            eps_annealing=0.05,
            **kwargs
    ):
        super(MPOT, self).__init__(
            name='MPOT',
            n_dof=n_dof,
            traj_len=traj_len,
            num_particles_per_goal=num_particles_per_goal,
            opt_iters=opt_iters,
            dt=dt,
            start_state=start_state,
            initial_particle_means=initial_particle_means,
            multi_goal_states=multi_goal_states,
            sigma_start_init=sigma_start_init,
            sigma_goal_init=sigma_goal_init,
            sigma_gp_init=sigma_gp_init,
            pos_only=False,
            **kwargs
        )
        self.d_state_opt = 2 * self.n_dof

        self.goal_directed = (multi_goal_states is not None)
        if not self.goal_directed:
            self.num_goals = 1
        else:
            assert multi_goal_states.dim() == 2
            self.num_goals = multi_goal_states.shape[0]

        self.num_bpoint = num_bpoint
        self.step_size = step_size
        self.probe_radius = probe_radius
        self.solver_params = solver_params
        self.solver_params['numItermax'] = opt_iters
        self.start_state = start_state
        self.multi_goal_states = multi_goal_states
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.random_init = random_init
        self.polytope = polytope
        self.random_step = random_step
        self.annealing = annealing
        self.eps_annealing = eps_annealing
        self._traj_dist = None

        ##############################################
        # Construct cost function
        self.cost = build_mpot_cost_composite(
            robot=robot,
            traj_len=traj_len,
            dt=dt,
            start_state=start_state,
            multi_goal_states=multi_goal_states,
            num_particles_per_goal=num_particles_per_goal,
            **kwargs
        )
        ##############################################
        # Initialize particles
        self.reset(initial_particle_means=initial_particle_means)

        # scaling operations
        if isinstance(pos_limits, torch.Tensor):
            self.pos_limits = pos_limits.clone().to(**self.tensor_args)
        else:
            self.pos_limits = torch.tensor(pos_limits, **self.tensor_args)
        if self.pos_limits.ndim == 1:
            self.pos_limits = self.pos_limits.unsqueeze(0).repeat(self.n_dof, 1)
        self.pos_scaler = MinMaxCenterScaler(dim_range=[0, self.n_dof], min=self.pos_limits[:, 0], max=self.pos_limits[:, 1])
        if isinstance(vel_limits, torch.Tensor):
            self.vel_limits = vel_limits.clone().to(**self.tensor_args)
        else:
            self.vel_limits = torch.tensor(vel_limits, **self.tensor_args)
        if self.vel_limits.ndim == 1:
            self.vel_limits = self.vel_limits.unsqueeze(0).repeat(self.n_dof, 1)
        self.vel_scaler = MinMaxCenterScaler(dim_range=[self.n_dof, self.d_state_opt], min=self.vel_limits[:, 0], max=self.vel_limits[:, 1])

    def reset(
            self,
            start_state=None,
            multi_goal_states=None,
            initial_particle_means=None,
    ):
        if start_state is not None:
            self.start_state = start_state.detach().clone()

        if multi_goal_states is not None:
            self.multi_goal_states = multi_goal_states.detach().clone()

        self.get_prior_samples(initial_particle_means=initial_particle_means)

    def get_prior_samples(self, initial_particle_means=None):
        origin = torch.zeros(self.d_state_opt, **self.tensor_args)
        if self.random_step:
            self.step_dist = None
            self.step_weights = None
        else:
            self.step_dist = POLYTOPE_MAP[self.polytope](origin, self.step_size, num_points=self.num_bpoint)
            self.step_weights = torch.ones(self.step_dist.shape[0], **self.tensor_args) / self.step_dist.shape[0]

        if initial_particle_means is None:
            if self.random_init:
                self.init_trajs = self.get_random_trajs()
            else:
                start, end = self.start_state.unsqueeze(0), self.multi_goal_states
                self.init_trajs = self.const_vel_trajectories(start, end)
                # copy traj for each particle
                self.init_trajs = self.init_trajs.unsqueeze(1).repeat(1, self.num_particles_per_goal, 1, 1)
                self.traj_dim = self.init_trajs.shape
                self.init_trajs = self.init_trajs.flatten(0, 1)
        else:
            self.init_trajs = initial_particle_means.clone()
        self.flatten_trajs = self.init_trajs.flatten(0, 1)

    def const_vel_trajectories(
        self,
        start_state,
        multi_goal_states,
    ):
        traj_dim = (multi_goal_states.shape[0], self.traj_len, self.d_state_opt)
        state_traj = torch.zeros(traj_dim, **self.tensor_args)
        mean_vel = (multi_goal_states[:, :self.n_dof] - start_state[:, :self.n_dof]) / (self.traj_len * self.dt)
        for i in range(self.traj_len):
            state_traj[:, i, :self.n_dof] = start_state[:, :self.n_dof] * (self.traj_len - i - 1) / (self.traj_len - 1) \
                                  + multi_goal_states[:, :self.n_dof] * i / (self.traj_len - 1)
        state_traj[:, :, self.n_dof:] = mean_vel.unsqueeze(1).repeat(1, self.traj_len, 1)
        return state_traj

    def get_dist(
            self,
            start_K,
            gp_K,
            goal_K,
            state_init,
            particle_means=None,
            goal_states=None,
    ):
        return MultiMPPrior(
            self.traj_len - 1,
            self.dt,
            self.d_state_opt,
            self.n_dof,
            start_K,
            gp_K,
            state_init,
            K_g_inv=goal_K,
            means=particle_means,
            goal_states=goal_states,
            tensor_args=self.tensor_args,
        )

    def get_random_trajs(self):
        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.n_dof * 2,
            self.sigma_start_init,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.n_dof,
            self.sigma_gp_init,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_init = []
        if self.goal_directed:
            for i in range(self.num_goals):
                self.multi_goal_prior_init.append(
                    UnaryFactor(
                        self.n_dof * 2,
                        self.sigma_goal_init,
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )
        self._traj_dist = self.get_dist(
                self.start_prior_init.K,
                self.gp_prior_init.Q_inv[0],
                self.multi_goal_prior_init[0].K if self.goal_directed else None,
                self.start_state,
                goal_states=self.multi_goal_states,
            )
        particles = self._traj_dist.sample(self.num_particles_per_goal)
        self.traj_dim = particles.shape
        del self._traj_dist  # free memory
        return particles.flatten(0, 1)

    def optimize(self, **kwargs) -> None:
        trajs, log_dict = optimize(self.step_dist, self.step_weights, self.flatten_trajs, self.cost, self.step_size, self.probe_radius,
                                   polytope=self.polytope, num_sphere_point=self.num_bpoint, annealing=self.annealing, eps_annealing=self.eps_annealing,
                                   traj_dim=self.traj_dim, pos_scaler=self.pos_scaler, vel_scaler=self.vel_scaler,
                                   **self.solver_params, **kwargs)
        trajs = trajs.view(self.traj_dim)
        return trajs, log_dict
