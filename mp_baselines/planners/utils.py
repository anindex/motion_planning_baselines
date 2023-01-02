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


def safe_path(sequence, collision):
    path = []
    for q in sequence:
        if collision(q):
            break
        path.append(q)
    return path


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
    path = to_numpy(path)
    diff = np.diff(path, axis=-2)
    # add last always, if same as second to last, second to last will not be selected
    cond = np.concatenate([np.any(diff > eps, axis=-1), [True]])
    reverse_doublicate_indices = np.where(cond)[0]
    selection = path[reverse_doublicate_indices]
    return selection
