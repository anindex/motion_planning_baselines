import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

from mp_baselines.planners.base import OptimizationPlanner


class CHOMP(OptimizationPlanner):

    def __init__(
            self,
            n_dof: int,
            traj_len: int,
            num_particles_per_goal: int,
            opt_iters: int,
            dt: float,
            start_state: torch.Tensor,
            cost=None,
            initial_particle_means=None,
            step_size: float = 1.,
            grad_clip: float = .01,
            multi_goal_states: torch.Tensor = None,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=10.,
            pos_only: bool = True,
            tensor_args: dict = None,
            **kwargs
    ):
        super(CHOMP, self).__init__(name='CHOMP',
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
                                    pos_only=pos_only,
                                    tensor_args=tensor_args)

        # CHOMP params
        self.lr = step_size
        self.grad_clip = grad_clip

        self._particle_means = None
        # Precision matrix, shape: [ctrl_dim, traj_len, traj_len]
        self.Sigma_inv = self._get_R_mat()
        self.Sigma = torch.inverse(self.Sigma_inv)
        self.reset(initial_particle_means=initial_particle_means)

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
        CHOMP time-correlated Precision matrix.
        """
        lower_diag = -torch.diag(torch.ones(self.traj_len - 1), diagonal=-1)
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

    def reset(
            self,
            initial_particle_means=None,
    ):
        # Straightline position-trajectory from start to goal with const vel
        if initial_particle_means is not None:
            self._particle_means = initial_particle_means.clone()
        else:
            self._particle_means = self.get_random_trajs()

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
            opt_iters = self.opt_iters
        for opt_step in range(opt_iters):
            self._particle_means.requires_grad_(True)
            costs = self._eval(self._particle_means, **observation)

            # Get grad
            costs.sum().backward(retain_graph=True)
            self._particle_means.grad.data.clamp_(-self.grad_clip, self.grad_clip)  # For stabilizing and preventing too high gradients
            # zeroing grad at start and goal
            self._particle_means.grad.data[..., 0, :] = 0.
            self._particle_means.grad.data[..., -1, :] = 0.

            # Update trajectory
            self._particle_means.data.add_(-self.lr * self._particle_means.grad.data)
            self._particle_means.grad.detach_()
            self._particle_means.grad.zero_()

        self._particle_means = self._particle_means.detach()

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
        costs += smooth_cost.diagonal(dim1=1, dim2=2).sum()

        return costs
