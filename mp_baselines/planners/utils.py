import time

import einops
import numpy as np
import torch
from scipy import interpolate


def elapsed_time(start_time):
    return time.time() - start_time

def extend_path(distance_fn, q1, q2, max_step=0.03, max_dist=0.1, tensor_args=None):
    # max_dist must be <= radius of RRT star!
    dist = distance_fn(q1, q2)
    if dist > max_dist:
        q2 = q1 + (q2 - q1) * (max_dist / dist)

    alpha = torch.linspace(0, 1, int(dist / max_step) + 2).to(**tensor_args)  # skip first and last
    q1 = q1.unsqueeze(0)
    q2 = q2.unsqueeze(0)
    extension = q1 + (q2 - q1) * alpha.unsqueeze(1)
    return extension


def safe_path(sequence, collision):
    in_collision = collision(sequence)
    idxs_in_collision = torch.argwhere(in_collision)
    if idxs_in_collision.nelement() == 0:
        if sequence.ndim == 1:
            return sequence.reshape((1, -1))
        return sequence[-1].reshape((1, -1))
    else:
        first_idx_in_collision = idxs_in_collision[0]
        if first_idx_in_collision == 0:
            # the first point in the sequence is in collision
            return []
        # return the point immediate before the one in collision
        return sequence[first_idx_in_collision-1]


def purge_duplicates_from_traj(path, eps=1e-6):
    # Remove duplicated points from a trajectory
    if len(path) < 2:
        return path
    if isinstance(path, list):
        path = torch.stack(path, dim=0)
    if path.shape[0] == 2:
        return path

    abs_diff = torch.abs(torch.diff(path, dim=-2))
    row_idxs = torch.argwhere(torch.all((abs_diff > eps) == False, dim=-1) == False).unique()
    selection = path[row_idxs]
    # Always add the first and last elements of the path
    if torch.allclose(selection[0], path[0]) is False:
        selection = torch.cat((path[0].view(1, -1), selection), dim=0)
    if torch.allclose(selection[-1], path[-1]) is False:
        selection = torch.cat((selection, path[-1].view(1, -1)), dim=0)
    return selection


def interpolate_traj_via_points(env, trajs, num_intepolation_traj=10):
    trajs = env.get_q_position(trajs)
    if num_intepolation_traj > 0:
        assert trajs.ndim > 1
        traj_dim = trajs.shape
        alpha = torch.linspace(0, 1, num_intepolation_traj + 2).type_as(trajs)[1:num_intepolation_traj + 1]
        alpha = alpha.view((1,) * len(traj_dim[:-1]) + (-1, 1))
        interpolated_trajs = trajs[..., 0:traj_dim[-2] - 1, None, :] * alpha + \
                             trajs[..., 1:traj_dim[-2], None, :] * (1 - alpha)
        interpolated_trajs = interpolated_trajs.view(traj_dim[:-2] + (-1, env.q_n_dofs))
    else:
        interpolated_trajs = trajs
    return interpolated_trajs


def get_collision_free_trajectories(trajs, env, return_per_viapoint=False, margin=0.001):
    trajs_new = trajs
    if trajs.ndim == 4:  # n_goals, batch of trajectories, length, dim
        trajs_new = einops.rearrange(trajs, 'n b l d -> (n b) l d')
    trajs_new = interpolate_traj_via_points(env, trajs_new)
    trajs_idxs_not_in_collision_via_points = torch.logical_not(env.compute_collision(trajs_new, margin=margin))
    if trajs.ndim == 4:
        trajs_idxs_not_in_collision_via_points = einops.rearrange(trajs_idxs_not_in_collision_via_points, '(n b) l -> n b l', n=trajs.shape[0])
    trajs_idxs_not_in_collision = trajs_idxs_not_in_collision_via_points.all(dim=-1)
    free_trajs_idxs = torch.argwhere(trajs_idxs_not_in_collision)
    if trajs.ndim == 4:
        free_trajs = trajs[free_trajs_idxs[:, 0], free_trajs_idxs[:, 1], :, :]
    else:
        free_trajs = trajs[free_trajs_idxs, :, :]
    if return_per_viapoint:
        return trajs_idxs_not_in_collision, free_trajs, trajs_idxs_not_in_collision_via_points
    return trajs_idxs_not_in_collision, free_trajs


def compute_percentage_collision_free_trajs(trajs, env):
    trajs_idxs_not_in_collision, _ = get_collision_free_trajectories(trajs, env)
    return (torch.count_nonzero(trajs_idxs_not_in_collision) / trajs.shape[0]).item()


def compute_collision_intensity_free_trajs(trajs, env):
    _, _, trajs_idxs_not_in_collision_via_points = get_collision_free_trajectories(trajs, env, return_per_viapoint=True)
    traj_len = trajs_idxs_not_in_collision_via_points.shape[-1]
    trajs_percentage_not_in_collision = torch.count_nonzero(trajs_idxs_not_in_collision_via_points, dim=-1) / traj_len
    trajs_percentage_in_collision = 1 - trajs_percentage_not_in_collision
    return trajs_percentage_in_collision


def success_collision_free_trajs(trajs, env):
    # if at least one trajectory is collision free, then we consider success
    trajs_idxs_not_in_collision, _ = get_collision_free_trajectories(trajs, env)
    count_trajs_not_in_collision = torch.count_nonzero(trajs_idxs_not_in_collision).item()
    if count_trajs_not_in_collision > 0:
        return 1
    else:
        return 0


def get_trajs_free_coll(trajs, env):
    trajs_idxs_not_in_collision, free_trajs = get_collision_free_trajectories(trajs, env)
    idxs_free = torch.argwhere(trajs_idxs_not_in_collision).squeeze()
    idxs_coll = torch.argwhere(torch.logical_not(trajs_idxs_not_in_collision)).squeeze()
    trajs_free = None
    trajs_coll = None
    if idxs_free.nelement() > 0:
        trajs_free = trajs[idxs_free]
        if trajs_free.ndim == 2:
            trajs_free = trajs_free.unsqueeze(0)
    if idxs_coll.nelement() > 0:
        trajs_coll = trajs[idxs_coll]
        if trajs_coll.ndim == 2:
            trajs_coll = trajs_coll.unsqueeze(0)
    return trajs_free, trajs_coll


def compute_path_length(trajs, env):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = env.get_q_position(trajs)
    path_length = torch.linalg.norm(torch.diff(trajs_pos, dim=-2), dim=-1).sum(-1)
    return path_length


def compute_smothness(trajs, env):
    assert trajs.ndim == 3
    if trajs.shape[-1] == env.q_n_dofs:
        # if there is no velocity information in the trajectory, compute it via finite difference
        trajs_pos = env.get_q_position(trajs)
        trajs_vel = torch.diff(trajs_pos, dim=-2)
    else:
        trajs_vel = env.get_q_velocity(trajs)
    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1).mean(-1)  # mean over trajectory horizon
    return smoothness


def smoothen_trajectory(traj, traj_len=30, zero_velocity=True, tensor_args=None):
    traj = to_numpy(traj)
    try:
        # bc_type='clamped' for zero velocities at start and finish
        spline_pos = interpolate.make_interp_spline(np.linspace(0, 1, traj.shape[0]), traj, k=3, bc_type='clamped')
        spline_vel = spline_pos.derivative(1)
    except TypeError:
        # Trajectory is too short to interpolate, so add last position again and interpolate
        traj = np.vstack((traj, traj[-1] + np.random.normal(0, 0.01)))
        return smoothen_trajectory(traj, traj_len=traj_len, tensor_args=tensor_args)

    pos = spline_pos(np.linspace(0, 1, traj_len))
    if zero_velocity:
        vel = np.zeros_like(pos)
    else:
        vel = spline_vel(np.linspace(0, 1, traj_len))
    return to_torch(pos, **tensor_args), to_torch(vel, **tensor_args)


@torch.jit.script
def tensor_linspace(start: torch.Tensor, end: torch.Tensor, steps: int = 10):
    # https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def to_torch(x, device='cpu', dtype=torch.float, requires_grad=False):
    if torch.is_tensor(x):
        return x.clone().to(device=device, dtype=dtype).requires_grad_(requires_grad)
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def to_numpy(x, dtype=np.float32):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().astype(dtype)
        return x
    return np.array(x).astype(dtype)


