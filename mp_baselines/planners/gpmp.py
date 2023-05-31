# Adapted from https://github.com/anindex/stoch_gpmp
import einops
import torch

from mp_baselines.planners.base import OptimizationPlanner
from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.mp_priors_multi import MultiMPPrior
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor


class GPMP(OptimizationPlanner):

    def __init__(
            self,
            n_dof: int,
            traj_len: int,
            num_particles_per_goal: int,
            opt_iters: int,
            dt: float,
            start_state: torch.Tensor,
            step_size=1.,
            multi_goal_states=None,
            initial_particle_means=None,
            cost=None,
            sigma_start_init=None,
            sigma_start_sample=None,
            sigma_goal_init=None,
            sigma_goal_sample=None,
            sigma_gp_init=None,
            sigma_gp_sample=None,
            solver_params=None,
            **kwargs
    ):
        super(GPMP, self).__init__(
            name='GPMP',
            n_dof=n_dof,
            traj_len=traj_len,
            num_particles_per_goal=num_particles_per_goal,
            opt_iters=opt_iters,
            dt=dt,
            start_state=start_state,
            cost=cost,
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

        self.step_size = step_size
        self.sigma_start_sample = sigma_start_sample
        self.sigma_goal_sample = sigma_goal_sample
        self.sigma_gp_sample = sigma_gp_sample

        self.solver_params = solver_params

        self.N = self.d_state_opt * self.traj_len

        self._mean = None
        self._weights = None
        self._dist = None

        self.reset(initial_particle_means=initial_particle_means)

    def set_prior_factors(self):

        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.d_state_opt,
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

        if self.goal_directed:
            self.multi_goal_prior_init = []
            for i in range(self.num_goals):
                self.multi_goal_prior_init.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_init,
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

        #========= Sampling factors ===============
        self.start_prior_sample = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_sample,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior_sample = GPFactor(
            self.n_dof,
            self.sigma_gp_sample,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        if self.goal_directed:
            self.multi_goal_prior_sample = []
            for i in range(self.num_goals):
                self.multi_goal_prior_sample.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_sample,
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

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
            2 * self.n_dof,
            self.n_dof,
            start_K,
            gp_K,
            state_init,
            K_g_inv=goal_K,  # Assume same goal Cov. for now
            means=particle_means,
            goal_states=goal_states,
            tensor_args=self.tensor_args,
        )

    def reset(
            self,
            start_state=None,
            multi_goal_states=None,
            initial_particle_means=None,
    ):

        if start_state is not None:
            self.start_state = start_state.clone()

        if multi_goal_states is not None:
            self.multi_goal_states = multi_goal_states.clone()

        self.set_prior_factors()

        # Initialization particles from prior distribution
        if initial_particle_means is not None:
            self._particle_means = initial_particle_means
        else:
            self._init_dist = self.get_dist(
                self.start_prior_init.K,
                self.gp_prior_init.Q_inv[0],
                self.multi_goal_prior_init[0].K if self.goal_directed else None,
                self.start_state,
                goal_states=self.multi_goal_states,
            )
            self._particle_means = self._init_dist.sample(self.num_particles_per_goal).to(**self.tensor_args)
            del self._init_dist  # freeing memory
        self._particle_means = self._particle_means.flatten(0, 1)

        self._sample_dist = self.get_dist(
            self.start_prior_sample.K,
            self.gp_prior_sample.Q_inv[0],
            self.multi_goal_prior_sample[0].K if self.goal_directed else None,
            self.start_state,
            particle_means=self._particle_means,
        )

    def optimize(
            self,
            opt_iters=None,
            debug=False,
            **observation
    ):

        if opt_iters is None:
            opt_iters = self.opt_iters

        for opt_step in range(opt_iters):
            b, K = self._step(**observation)

        self.costs = self._get_costs(b, K)

        position_seq_mean = self._particle_means[..., :self.n_dof].clone()
        velocity_seq_mean = self._particle_means[..., -self.n_dof:].clone()
        # costs = self.costs.clone()

        self._recent_state_trajectories = position_seq_mean
        self._recent_control_particles = velocity_seq_mean

        # get mean trajectory
        curr_traj = self._get_traj()
        return curr_traj

    def _step(self, **observation):
        A, b, K = self.cost.get_linear_system(self._particle_means, **observation)

        J_t_J, g = self._get_grad_terms(
            A, b, K,
            delta=self.solver_params['delta'],
            trust_region=self.solver_params['trust_region'],
        )

        d_theta = self.get_torch_solve(
            J_t_J, g,
            method=self.solver_params['method'],
        )

        d_theta = d_theta.view(
                self.num_particles,
                self.traj_len,
                self.d_state_opt,
            )

        self._particle_means = self._particle_means + self.step_size * d_theta
        self._particle_means.detach_()

        return b, K

    def _get_grad_terms(
            self,
            A, b, K,
            delta=0.,
            trust_region=False,
    ):
        I = torch.eye(self.N, self.N, **self.tensor_args)
        A_t_K = A.transpose(-2, -1) @ K
        A_t_A = A_t_K @ A
        if not trust_region:
            J_t_J = A_t_A + delta * I
        else:
            # J_t_J = A_t_A + delta * I * torch.diagonal(A_t_A, dim1=-2, dim2=-1).unsqueeze(-1)
            # Since hessian will be averaged over particles, add diagonal matrix of the mean.
            diag_A_t_A = A_t_A.mean(0) * I
            J_t_J = A_t_A + delta * diag_A_t_A
        g = A_t_K @ b
        return J_t_J, g

    def get_torch_solve(
        self,
        A, b,
        method,
    ):
        if method == 'inverse':
            return torch.linalg.solve(A, b)
        elif method == 'cholesky':
            # method 1
            # old implementation - recheck torch.allclose(res, torch.linalg.solve(A, b))
            # l = torch.linalg.cholesky(A)
            # z = torch.linalg.solve_triangular(l, b, upper=False)
            # res = torch.linalg.solve_triangular(l.mT, z, upper=False)

            # method 2
            # z = torch.triangular_solve(b, l, transpose=False, upper=False)[0]
            # res = torch.triangular_solve(z, l, transpose=True, upper=False)[0]

            # method 3
            l, _ = torch.linalg.cholesky_ex(A)
            res = torch.cholesky_solve(b, l)

            return res

        else:
            raise NotImplementedError

    def _get_costs(self, errors, w_mat):
        costs = errors.transpose(1, 2) @ w_mat.unsqueeze(0) @ errors
        return costs.reshape(self.num_particles,)

    def get_recent_samples(self):
        vel = self._recent_control_particles.detach().clone()
        vel = einops.rearrange(vel, '(m b) h d -> m b h d', m=self.num_goals)
        pos = self._recent_state_trajectories.detach().clone()
        pos = einops.rearrange(pos, '(m b) h d -> m b h d', m=self.num_goals)
        # pos_mean = self._particle_means[..., :self.n_dof].detach().clone()
        # vel_mean = self._particle_means[..., -self.n_dof:].detach().clone()

        return (
            pos,
            vel,
        )
