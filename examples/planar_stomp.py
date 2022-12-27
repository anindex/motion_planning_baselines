import time
import torch
import random
import matplotlib.pyplot as plt

from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from torch_planning_objectives.fields.shape_distance_fields import MultiSphere
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite
from mp_baselines.planners.stomp import STOMP


if __name__ == "__main__":
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float64}

    n_dof = 2
    traj_len = 64
    dt = 0.1
    num_particles_per_goal = 5
    num_samples = 32
    seed = 11
    start_q = torch.tensor([-9, -9], **tensor_args)
    start_state = torch.cat((start_q, torch.zeros(2, **tensor_args)))
    goal_q = torch.tensor([9, 8], **tensor_args)
    goal_state = torch.cat((goal_q, torch.zeros(2, **tensor_args)))

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
        rand_shape=[2, 2],
        tensor_args=tensor_args,
    )
    # For obst. generation
    random.seed(seed)
    obst_map, obst_list = generate_obstacle_map(**obst_params)

    #-------------------------------- Cost func. ---------------------------------
    sigma_coll = 1e-4

    # Construct cost function
    cost_obst_2D = CostCollision(n_dof, traj_len, field=obst_map, sigma_coll=sigma_coll)
    cost_func_list = [cost_obst_2D]
    cost_composite = CostComposite(n_dof, traj_len, cost_func_list)

    ## Planner - 2D point particle dynamics
    stomp_params = dict(
        n_dofs=n_dof,
        traj_len=traj_len,
        num_samples=num_samples,
        n_iters=1, # Keep this 1 for visualization
        dt=dt,
        init_q=start_q,
        cost=cost_composite,
        temperature=1.,
        step_size=0.5,
        grad_clip=0.1,
        goal_state=goal_state,
        pos_only=False,
        tensor_args=tensor_args,
    )
    planner = STOMP(**stomp_params)

    #---------------------------------------------------------------------------
    # Optimize
    # opt_iters = 50
    opt_iters = 50
    # opt_iters = 1000

    traj_history = []
    sample_history = []
    for i in range(opt_iters + 1):
        print(i)
        time_start = time.time()
        trajectories = planner.optimize()
        print(time.time() - time_start)
        traj_history.append(trajectories)
        sample_history.append(planner.state_particles.clone())
    #---------------------------------------------------------------------------
    # Plotting

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
        ax.set_aspect('equal')
        trajs = trajs.cpu().numpy()
        for i in range(trajs.shape[0]):
            ax.plot(trajs[i, :, 0], trajs[i, :, 1], 'r-', markersize=3)
        traj_mean = traj_history[iter].cpu().numpy()
        ax.plot(traj_mean[:, 0], traj_mean[:, 1], 'b-', markersize=3)
        plt.draw()
        plt.pause(0.01)
