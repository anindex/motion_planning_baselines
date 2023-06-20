import torch
import numpy as np
from itertools import product

from .probe import get_random_probe_points, get_probe_points
from .rotation import get_random_maximal_torus_matrix


def get_cube_vertices(origin, radius=1., **kwargs):
    dim = origin.shape[-1]
    points = torch.tensor(list(product([1, -1], repeat=dim))) / np.sqrt(dim)
    points = points.type_as(origin) * radius + origin
    return points


def get_sampled_cube_vertices(origin, polytope_vertices=None, step_radius=1., probe_radius=2., num_probe=5, random_probe=False, **kwargs):
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape
    if polytope_vertices is None:
        polytope_vertices = torch.tensor(list(product([1, -1], repeat=dim))).type_as(origin) / np.sqrt(dim)
    polytope_vertices = polytope_vertices.unsqueeze(0).repeat(batch, 1, 1)  # [batch, 2 ^ dim, dim]
  
    max_torus_mat = get_random_maximal_torus_matrix(origin, angle_range=[0, np.pi/2])
    polytope_vertices = torch.matmul(polytope_vertices, max_torus_mat)
    step_points = polytope_vertices * step_radius + origin.unsqueeze(1)  # [batch, 2 ^ dim, dim]
    if random_probe:
        probe_points = get_random_probe_points(origin, polytope_vertices, probe_radius, num_probe)
    else:
        probe_points = get_probe_points(origin, polytope_vertices, probe_radius, num_probe)  # [batch, 2 ^ dim, num_probe, dim]
    return step_points, probe_points, polytope_vertices


def get_orthoplex_vertices(origin, radius=1., **kwargs):
    dim = origin.shape[-1]
    points = torch.zeros((2 * dim, dim)).type_as(origin)
    first = torch.arange(0, dim)
    second = torch.arange(dim, 2 * dim)
    points[first, first] = radius
    points[second, first] = -radius
    points = points + origin
    return points


def get_sampled_orthoplex_vertices(origin, polytope_vertices=None, step_radius=1., probe_radius=2., num_probe=5, random_probe=False, **kwargs):
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape
    if polytope_vertices is None:
        polytope_vertices = torch.zeros((2 * dim, dim)).type_as(origin)
        first = torch.arange(0, dim)
        second = torch.arange(dim, 2 * dim)
        polytope_vertices[first, first] = 1
        polytope_vertices[second, first] = -1
    polytope_vertices = polytope_vertices.unsqueeze(0).repeat(batch, 1, 1)  # [batch, 2 * dim, dim]

    max_torus_mat = get_random_maximal_torus_matrix(origin, angle_range=[0, np.pi/2])
    polytope_vertices = torch.matmul(polytope_vertices, max_torus_mat)
    step_points = polytope_vertices * step_radius + origin.unsqueeze(1)  # [batch, 2 * dim, dim]
    if random_probe:
        probe_points = get_random_probe_points(origin, polytope_vertices, probe_radius, num_probe)
    else:
        probe_points = get_probe_points(origin, polytope_vertices, probe_radius, num_probe)  # [batch, 2 * dim, num_probe, dim]
    return step_points, probe_points, polytope_vertices


def get_simplex_vertices(origin, radius=1., **kwargs):
    dim = origin.shape[-1]
    points = torch.zeros(dim, dim + 1).type_as(origin)
    for k in range(dim):
        # set X(K,K) so that sum ( X(1:K,K)^2 ) = 1.
        s = torch.square(points[:k, k]).sum()
        points[k, k] = torch.sqrt(1 - s)
        # set X(K,J) for J = K+1 to M+1 by using the fact that XK dot XJ = - 1 / M.
        for j in range(k + 1, dim + 1):
            s = 0.0
            for i in range(k):
                s = s + points[i, k] * points[i, j]
            points[k, j] = (-1 / dim - s) / points[k, k]
    points = points.T * radius + origin
    return points


def get_sampled_simplex_vertices(origin, polytope_vertices=None, step_radius=1., probe_radius=2., num_probe=5, random_probe=False, **kwargs):
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape
    if polytope_vertices is None:
        points = torch.zeros(dim, dim + 1).type_as(origin)
        for k in range(dim):
            # set X(K,K) so that sum ( X(1:K,K)^2 ) = 1.
            s = torch.square(points[:k, k]).sum()
            points[k, k] = torch.sqrt(1 - s)
            # set X(K,J) for J = K+1 to M+1 by using the fact that XK dot XJ = - 1 / M.
            for j in range(k + 1, dim + 1):
                s = 0.0
                for i in range(k):
                    s = s + points[i, k] * points[i, j]
                points[k, j] = (-1 / dim - s) / points[k, k]
        polytope_vertices = points.T
    polytope_vertices = polytope_vertices.unsqueeze(0).repeat(batch, 1, 1)  # [batch, 2 * dim, dim]

    max_torus_mat = get_random_maximal_torus_matrix(origin, angle_range=[0, 2 / 3 * np.pi])
    polytope_vertices = torch.matmul(polytope_vertices, max_torus_mat)
    step_points = polytope_vertices * step_radius + origin.unsqueeze(1)  # [batch, 2 * dim, dim]
    if random_probe:
        probe_points = get_random_probe_points(origin, polytope_vertices, probe_radius, num_probe)
    else:
        probe_points = get_probe_points(origin, polytope_vertices, probe_radius, num_probe)  # [batch, 2 * dim, num_probe, dim]
    return step_points, probe_points, polytope_vertices


def get_sampled_points_on_sphere(origin, step_radius=1., probe_radius=2., num_probe=5, num_sphere_point=50, random_probe=False, **kwargs):
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape
    # marsaglia method
    points = torch.randn(batch, num_sphere_point, dim).type_as(origin)  # [batch, num_points, dim]
    points = points / points.norm(dim=-1, keepdim=True)
    step_points = points * step_radius + origin.unsqueeze(1)  # [batch, num_points, dim]
    if random_probe:
        probe_points = get_random_probe_points(origin, points, probe_radius, num_probe)
    else:
        probe_points = get_probe_points(origin, points, probe_radius, num_probe)  # [batch, 2 * dim, num_probe, dim]
    return step_points, probe_points, points


POLYTOPE_MAP = {
    'cube': get_cube_vertices,
    'orthoplex': get_orthoplex_vertices,
    'simplex': get_simplex_vertices,
}


SAMPLE_POLYTOPE_MAP = {
    'cube': get_sampled_cube_vertices,
    'orthoplex': get_sampled_orthoplex_vertices,
    'simplex': get_sampled_simplex_vertices,
    'random': get_sampled_points_on_sphere,
}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    points, probe_points, polytope_vertices = get_sampled_simplex_vertices(torch.zeros(2), step_radius=0.2, probe_radius=0.5, num_probe=5, random_probe=False)
    points = points.view(-1, 2)
    probe_points = probe_points.view(-1, 2)
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.scatter(points[:, 0], points[:, 1], c='r')
    ax.scatter(probe_points[:, 0], probe_points[:, 1], c='b')
    plt.show()
