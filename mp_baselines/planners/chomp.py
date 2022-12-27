import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

from mp_baselines.planners.base import MPPlanner


class CHOMP(MPPlanner):

    def __init__(
            self,
            n_dofs: int,
            traj_len: int,
            num_particles: int,
            n_iters: int,
            dt: float,
            init_q: torch.Tensor,
            cost=None,
            temperature: float = 1.,
            step_size: float = 1.,
            grad_clip: float = .01,
            goal_state_init: torch.Tensor = None,
            pos_only: bool = False,
            tensor_args: dict = None
    ):
        super(CHOMP, self).__init__(name='CHOMP', tensor_args=tensor_args)
        self.n_dofs = n_dofs
        self.traj_len = traj_len
        self.n_iters = n_iters
        self.pos_only = pos_only
        self.dt = dt

        # CHOMP params
        self.lr = step_size
        self.grad_clip = grad_clip

        self.init_q = init_q
        self.goal_state_init = goal_state_init
        self.num_particles = num_particles
        self.temperature = temperature

        self.cost = cost

        if self.pos_only:
            self.d_state_opt = self.n_dofs
            self.start_state = self.init_q
        else:
            self.d_state_opt = 2 * self.n_dofs
            self.start_state = torch.cat([self.init_q, torch.zeros_like(self.init_q)], dim=-1)
            self.goal_state_init = torch.cat([self.goal_state_init, torch.zeros_like(self.goal_state_init)], dim=-1)

        self._mean = None
        self._weights = None
        self._sample_dist = None

        # Precision matrix, shape: [ctrl_dim, traj_len, traj_len]
        self.Sigma_inv = self._get_R_mat()
        self.Sigma = torch.inverse(self.Sigma_inv)
        self.reset()

    def _get_R_mat2(self):
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

    def _get_R_mat(self):
        """
        STOMP time-correlated Precision matrix.
        """
        lower_diag = -1*torch.diag(torch.ones(self.traj_len - 1), diagonal=-1, )
        diag = 1 * torch.eye(self.traj_len)
        A_mat = diag + lower_diag
        A_mat = torch.cat(
            (A_mat,
             torch.zeros(1, self.traj_len)),
            dim=0,
        )
        A_mat[-1, -1] = -1.
        # A_mat[-1, -1] = 1.
        A_mat = A_mat * 1. / self.dt ** 2
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
            Return: position-trajectory samples, of shape: [num_particles, traj_len, n_dof]
        """
        noise = self._noise_dist.sample((self.num_particles, self.d_state_opt)).transpose(1, 2)

        # Force Bound to zero ##
        noise[:, -1, :] = torch.zeros_like(noise[:, -1, :])
        noise[:, 0, :] = torch.zeros_like(noise[:, 0, :])

        samples = self._mean.unsqueeze(0) + noise
        return samples

    def reset(
            self,
            start_state=None,
            goal_state=None,
    ):
        self._reset_traj(start_state, goal_state)

    def _reset_traj(
            self,
            start_state=None,
            goal_state=None,
    ):

        if start_state is None:
            start_state = self.start_state.clone()

        if goal_state is None:
            goal_state = self.goal_state_init.clone()

        # Straightline position-trajectory from start to goal with const vel
        self._mean = self.const_vel_trajectory(start_state, goal_state)

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
            self._mean.requires_grad_(True)
            costs = self._eval(self._mean, **observation)

            # Get grad
            costs.sum().backward(retain_graph=True)
            self._mean.grad.data.clamp_(-self.grad_clip, self.grad_clip)  # For stabilizing and preventing too high gradients
            # zeroing grad at start and goal
            self._mean.grad.data[0, :] = torch.zeros_like(self._mean.grad.data[0, :])
            self._mean.grad.data[-1, :] = torch.zeros_like(self._mean.grad.data[-1, :])

            # Update trajectory
            self._mean.data.add_(-self.lr * self._mean.grad.data)
            self._mean.grad.detach_()
            self._mean.grad.zero_()

            if optim_vis:
                # print(opt_step)
                trj = self._mean.detach().cpu().numpy()
                plt.figure(1)
                plt.clf()
                plt.plot(trj[:, 0], trj[:, 1])
                plt.plot(trj[:, 0], trj[:, 1], 'o')
                plt.draw()
                plt.pause(0.05)

        self._mean = self._mean.detach()

    def _eval(self, x, **observation):
        """
            Evaluate costs.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # Evaluate quadratic costs
        costs = self._get_costs(x, **observation)

        # Add smoothness term
        R_mat = self.Sigma_inv.clone()
        smooth_cost = x.transpose(1, 2) @ R_mat.unsqueeze(0) @ x
        costs += 1 * smooth_cost.diagonal(dim1=1, dim2=2).sum()

        return costs

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
            costs = torch.zeros(self.num_particles, )
        else:
            costs = self.cost.eval(state_trajectories, **observation)
        return costs
