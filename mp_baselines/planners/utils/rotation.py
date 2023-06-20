import torch
from torch._vmap_internals import _vmap


def rotation_matrix(theta):
    theta = theta.unsqueeze(1).unsqueeze(1)
    dim1 = torch.cat([torch.cos(theta), -torch.sin(theta)], dim=2)
    dim2 = torch.cat([torch.sin(theta), torch.cos(theta)], dim=2)
    mat = torch.cat([dim1, dim2], dim=1)
    return mat


def get_random_maximal_torus_matrix(origin, angle_range=[0, torch.pi/2], **kwargs):
    batch, dim = origin.shape
    assert dim % 2 == 0, 'Only work with even dim for random rotation for now.'
    theta = torch.rand(dim // 2, batch).type_as(origin) * (angle_range[1] - angle_range[0]) + angle_range[0]  # [batch, dim // 2]
    rot_mat = _vmap(rotation_matrix)(theta).transpose(0, 1)
    # make batch block diag
    max_torus_mat = torch.diag_embed(rot_mat[:, :, [0, 1], [0, 1]].flatten(-2, -1), offset=0)
    even, odd = torch.arange(0, dim, 2), torch.arange(1, dim, 2)
    max_torus_mat[:, even, odd] = rot_mat[:, :, 0, 1]
    max_torus_mat[:, odd, even] = rot_mat[:, :, 1, 0]
    return max_torus_mat
