import torch


class FieldFactor:

    def __init__(
            self,
            n_dof,
            sigma,
            traj_range,
    ):
        self.sigma = sigma
        self.n_dof = n_dof
        self.traj_range = traj_range
        self.length = traj_range[1] - traj_range[0]
        self.K = 1. / (sigma**2)

    def get_error(
            self,
            q_trajs,
            field,
            q_pos=None,
            q_vel=None,
            H_pos=None,
            calc_jacobian=True,
            **kwargs
    ):
        batch = q_trajs.shape[0]

        if H_pos is not None:
            states = H_pos[:, self.traj_range[0]:self.traj_range[1]]
        else:
            states = q_trajs[:, self.traj_range[0]:self.traj_range[1], :self.n_dof].reshape(-1, self.n_dof)

        q_pos_new = q_pos[:, self.traj_range[0]:self.traj_range[1], :]

        error = field.compute_cost(q_pos_new, states, **kwargs).reshape(batch, self.length)

        if calc_jacobian:
            H = -torch.autograd.grad(error.sum(), q_trajs, retain_graph=True)[0][:, self.traj_range[0]:self.traj_range[1], :self.n_dof]
            error = error.detach()
            field.zero_grad()
            return error, H
        else:
            return error
