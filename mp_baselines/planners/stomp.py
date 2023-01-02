import torch
import torch.distributions as dist

from mp_baselines.planners.base import MPPlanner


class STOMP(MPPlanner):

    def __init__(
            self,
            n_dofs: int,
            traj_len: int,
            num_samples: int,
            n_iters: int,
            dt: float,
            start_state: torch.Tensor,
            cost=None,
            initial_mean=None,
            temperature: float = 1.,
            step_size: float = 1.,
            grad_clip: float = .01,
            goal_state: torch.Tensor = None,
            pos_only: bool = True,
            tensor_args: dict = None
    ):
        super(STOMP, self).__init__(name='STOMP', tensor_args=tensor_args)
        self.n_dofs = n_dofs
        self.traj_len = traj_len
        self.n_iters = n_iters
        self.pos_only = pos_only
        self.dt = dt

        # STOMP params
        self.lr = step_size
        self.grad_clip = grad_clip

        self.start_state = start_state
        self.goal_state = goal_state
        self.num_samples = num_samples
        self.temperature = temperature

        self.cost = cost

        if self.pos_only:
            self.d_state_opt = self.n_dofs
            self.start_state = self.start_state
        else:
            self.d_state_opt = 2 * self.n_dofs
            self.start_state = torch.cat([self.start_state, torch.zeros_like(self.start_state)], dim=-1)
            self.goal_state = torch.cat([self.goal_state, torch.zeros_like(self.goal_state)], dim=-1)

        self._mean = None
        self._weights = None
        self._sample_dist = None

        # Precision matrix, shape: [ctrl_dim, traj_len, traj_len]
        self.Sigma_inv = self._get_R_mat()
        self.Sigma = torch.inverse(self.Sigma_inv)
        self.reset(initial_mean=initial_mean)
        self.best_cost = torch.inf

    def _get_R_mat(self):
        """
        STOMP time-correlated Precision matrix.
        """
        upper_diag = torch.diag(torch.ones(self.traj_len - 1), diagonal=1)
        lower_diag = torch.diag(torch.ones(self.traj_len - 1), diagonal=-1,)
        diag = -2 * torch.eye(self.traj_len)
        A_mat = upper_diag + diag + lower_diag
        A_mat = torch.cat(
            (torch.zeros(1, self.traj_len),
             A_mat,
             torch.zeros(1, self.traj_len)),
            dim=0,
        )
        A_mat[0, 0] = 1.
        A_mat[-1, -1] = 1.
        A_mat = A_mat * 1./self.dt**2
        R_mat = A_mat.t() @ A_mat
        return R_mat.to(**self.tensor_args)

    def set_noise_dist(self):
        """
            Additive Gaussian noise distribution over 1-dimensional trajectory.
        """
        self._noise_dist = dist.MultivariateNormal(
            torch.zeros(self.traj_len, **self.tensor_args),
            precision_matrix=self.Sigma_inv,
        )

    def sample(self):
        """
            Generate trajectory samples from Gaussian control dist.
            Return: position-trajectory samples, of shape: [num_samples, traj_len, n_dof]
        """
        noise = self._noise_dist.sample((self.num_samples, self.d_state_opt)).transpose(1, 2)

        # Force Bound to zero ##
        noise[:, -1, :] = 0
        noise[:, 0, :] = 0

        samples = self._mean.unsqueeze(0) + noise
        return samples

    def reset(
            self,
            start_state=None,
            goal_state=None,
            initial_mean=None,
    ):
        if start_state is None:
            start_state = self.start_state.clone()

        if goal_state is None:
            goal_state = self.goal_state.clone()

        # Straightline position-trajectory from start to goal
        if initial_mean is not None:
            self._mean = initial_mean.clone()
        else:
            self._mean = self.const_vel_trajectory(start_state, goal_state)
        self._mean = self.const_vel_trajectory(start_state, goal_state)
        self.set_noise_dist()
        self.state_particles = self.sample()

    def const_vel_trajectory(
        self,
        start_state,
        goal_state,
    ):
        num_steps = self.traj_len - 1
        state_traj = torch.zeros(num_steps + 1, self.d_state_opt, **self.tensor_args)
        for i in range(num_steps + 1):
            state_traj[i, :self.n_dofs] = start_state[:self.n_dofs] * (num_steps - i) * 1. / num_steps \
                                  + goal_state[:self.n_dofs] * i * 1./num_steps
        if not self.pos_only:
            mean_vel = (goal_state[:self.n_dofs] - start_state[:self.n_dofs]) / (num_steps * self.dt)
            state_traj[:, self.n_dofs:] = mean_vel.unsqueeze(0)
        return state_traj

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):
        """
        Optimize for best trajectory at current state
        """

        self._run_optimization(opt_iters, **observation)
        # get best trajectory
        curr_traj = self._get_traj()
        return curr_traj

    def _run_optimization(self, opt_iters, **observation):
        """
            Run optimization iterations.
        """
        optim_vis = observation.get('optim_vis', False)
        if opt_iters is None:
            opt_iters = self.n_iters
        for opt_step in range(opt_iters):
            self.costs = self._sample_and_eval()
            self._update_distribution(self.costs, self.state_particles)
            self._mean = self._mean.detach()

    def _sample_and_eval(self, **observation):
        """
            Sample trajectories from distribution and evaluate costs.
        """
        # Sample state-trajectory particles
        self.state_particles = self.sample()

        ## Optional : Clamp for joint limits
        # if isinstance(self.max_control, list):
        #     for d in range(self.control_dim):
        #         control_samples[..., d] = control_samples[..., d].clamp(
        #             min=-self.max_control[d],
        #             max=self.max_control[d],
        #         )
        # else:
        #     control_samples = control_samples.clamp(
        #         min=-self.max_control,
        #         max=self.max_control,
        #     )

        # Evaluate quadratic costs
        costs = self._get_costs(self.state_particles, **observation)

        ## Optional : Add cost term from importance sampling ratio
        # for i in range(self.control_dim):
        #     Sigma_inv = self.ctrl_dist.Sigma[..., i].inverse()
        #     V = self.state_particles[..., i]
        #     U = self.U_mean[..., i]
        #     costs += self.lambda_ * (V @ Sigma_inv @ U).reshape(-1, 1)

        # Add smoothness term
        # smooth_cost = self.state_particles.transpose(1, 2) @ self.Sigma_inv.unsqueeze(0) @ self.state_particles
        # costs += smooth_cost.diagonal(dim1=1, dim2=2).sum(-1)

        return costs

    def _update_distribution(self, costs, traj_particles):
        """
            Get sample weights and pdate trajectory mean.
        """
        self._weights = self._calc_sample_weights(costs)
        self._weights = self._weights.reshape(-1, 1, 1)

        ## Optional: STOMP Covariance-weighted update
        self._mean.add_(
            self.lr * self.Sigma @ (
                self._weights * (traj_particles - self._mean.unsqueeze(0))
            ).sum(0)
        )

        # self._mean.add_(
        #     self.lr * (
        #         self._weights * (traj_particles - self._mean.unsqueeze(0))
        #     ).sum(0)
        # )

    def _calc_sample_weights(self, costs):
        return torch.softmax( -costs / self.temperature, dim=0)

    def _get_traj(self):
        """
            Get position-velocity trajectory from control distribution.
        """
        traj = self._mean.clone()
        if self.pos_only:
            # Linear velocity by finite differencing
            vels = (traj[:-1] - traj[1:]) / self.dt
            # Pad end with zero-vel for planning
            vels = torch.cat(
                (vels,
                 torch.zeros(1, self.n_dofs, **self.tensor_args)),
                dim=0,
            )
            traj = torch.cat((traj, vels), dim=1)
        return traj

    def _get_costs(self, state_trajectories, **observation):
        if self.cost is None:
            costs = torch.zeros(self.num_samples, )
        else:
            costs = self.cost.eval(state_trajectories, **observation)
        return costs
