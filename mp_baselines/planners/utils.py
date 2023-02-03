import time
import numpy as np
import torch

def elapsed_time(start_time):
    return time.time() - start_time


def argmin(function, sequence):
    # TODO: use min
    scores = [function(n) for n in sequence]
    min_idx = min(enumerate(scores), key=lambda x: x[1])[0]
    return sequence[min_idx]


# def safe_path(sequence, collision):
#     path = []
#     for q in sequence:
#         if collision(q):
#             break
#         path.append(q)
#     return path

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


def purge_duplicates_from_traj(path, eps=1e-5):
    if len(path) < 2:
        return path
    if isinstance(path, list):
        path = torch.stack(path, dim=0)
    abs_diff = torch.abs(torch.diff(path, dim=-2))
    row_idxs = torch.where(abs_diff > eps)[0].unique()
    selection = path[row_idxs]
    # Always add the first and last elements of the path
    selection[0] = path[0]
    selection[-1] = path[-1]
    return selection


def get_collision_free_trajectories(trajs, obst_map):
    trajs_idxs_not_in_collision = 1 - obst_map.get_collisions(trajs)
    free_trajs_idxs = trajs_idxs_not_in_collision.all(dim=-1)
    free_trajs = trajs[free_trajs_idxs, :, :]
    return free_trajs_idxs, free_trajs
