import time
import numpy as np
import torch
from scipy import interpolate


def elapsed_time(start_time):
    return time.time() - start_time


def argmin(function, sequence):
    # TODO: use min
    scores = [function(n) for n in sequence]
    min_idx = min(enumerate(scores), key=lambda x: x[1])[0]
    return sequence[min_idx]


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


def to_numpy(x, dtype=np.float32):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().astype(dtype)
        return x
    return np.array(x).astype(dtype)


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


def get_collision_free_trajectories(trajs, obst_map):
    trajs_idxs_not_in_collision = 1 - obst_map.get_collisions(trajs)
    free_trajs_idxs = trajs_idxs_not_in_collision.all(dim=-1)
    free_trajs = trajs[free_trajs_idxs, :, :]
    return free_trajs_idxs, free_trajs


def smoothen_trajectory(traj, traj_len=30, tensor_args=None):
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


