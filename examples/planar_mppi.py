import time
import torch
import random
import matplotlib.pyplot as plt

from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from torch_planning_objectives.fields.shape_distance_fields import MultiSphere
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite
from mp_baselines.planners.mppi import MPPI
from mp_baselines.planners.dynamics.point import PointParticleDynamics


if __name__ == "__main__":
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float64}

    n_dof = 2
    traj_len = 64
    dt = 0.02
    num_samples = 32
    seed = int(time.time())
    start_state = torch.tensor([-9, -9], **tensor_args)
    goal_state = torch.tensor([9, 8], **tensor_args)

    ## Obstacle map
    # obst_list = [(0, 0, 4, 6)]
    obst_list = []
    cell_size = 0.1
    map_dim = [20, 20]

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        map_type='direct',
        random_gen=True,
        num_obst=10,
        rand_xy_limits=[[-7.5, 7.5], [-7.5, 7.5]],
        rand_rect_shape=[2, 2],
        tensor_args=tensor_args,
    )
    # For obst. generation
    random.seed(seed)
    obst_map, obst_list = generate_obstacle_map(**obst_params)

    mppi_params = dict(
        num_ctrl_samples=num_samples,
        rollout_steps=traj_len,
        control_std=[0.15, 0.15],
        temp=1.,
        opt_iters=1,
        step_size=1.,
        cov_prior_type='const_ctrl',
        tensor_args=tensor_args,
    )

    system_params = dict(
        rollout_steps=mppi_params['rollout_steps'],
        control_dim=n_dof,
        state_dim=n_dof,
        dt=dt,
        discount=1.,
        goal_state=goal_state,
        ctrl_min=[-100, -100],
        ctrl_max=[100, 100],
        verbose=True,
        c_weights={
            'pos': 1.,
            'vel': 1.,
            'ctrl': 1.,
            'pos_T': 1000.,
            'vel_T': 0.,
        },
        tensor_args=tensor_args,
    )
    system = PointParticleDynamics(**system_params)
    controller = MPPI(system, **mppi_params)

    #-------------------------------- Cost func. ---------------------------------
    sigma_coll = 1e-5

    # Construct cost function
    cost_obst_2D = CostCollision(n_dof, traj_len, field=obst_map, sigma_coll=sigma_coll)
    cost_func_list = [cost_obst_2D]
    cost_composite = CostComposite(n_dof, traj_len, cost_func_list)

    #---------------------------------------------------------------------------
    # Optimize
    # opt_iters = 50
    opt_iters = 200
    # opt_iters = 1000

    observation = {
        'state': start_state,
        'goal_state': goal_state,
        'cost': cost_composite,
    }

    sample_history = []
    for i in range(opt_iters + 1):
        print(i)
        time_start = time.time()
        _, trajectories, _ = controller.optimize(**observation)
        print(time.time() - time_start)
        sample_history.append(trajectories)
    #---------------------------------------------------------------------------
    # Plotting
    start = start_state.cpu().numpy()
    goal = goal_state.cpu().numpy()
    best_traj = controller.best_traj.cpu().numpy()
    import numpy as np
    res = 200
    x = np.linspace(-10, 10, res)
    y = np.linspace(-10, 10, res)
    X, Y = np.meshgrid(x,y)
    grid = torch.from_numpy(np.stack((X, Y), axis=-1)).to(**tensor_args)
    fig = plt.figure()
    for iter, trajs in enumerate(sample_history):
        plt.clf()
        ax = fig.gca()
        cs = ax.contourf(x, y, obst_map.map, 20, cmap='Greys')
        ax.plot(start[0], start[1], 'go', markersize=7)
        ax.plot(goal[0], goal[1], 'ro', markersize=7)
        ax.set_aspect('equal')
        trajs = trajs.cpu().numpy()
        for i in range(trajs.shape[0]):
            ax.plot(trajs[i, :, 0], trajs[i, :, 1], 'r-', markersize=3)
        traj_mean = trajs.mean(axis=0)
        ax.plot(traj_mean[:, 0], traj_mean[:, 1], 'b-', markersize=3)
        ax.plot(best_traj[:, 0], best_traj[:, 1], 'g-', markersize=3)
        plt.draw()
        plt.pause(0.01)
