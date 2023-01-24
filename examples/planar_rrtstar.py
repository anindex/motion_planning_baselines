import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np

from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite
from mp_baselines.planners.rrt import RRTStar


if __name__ == "__main__":
    # device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

    n_dof = 2
    opt_iters = 5000
    num_particle_per_goal = 5
    step_size = 1    
    n_radius = 2.
    goal_prob = 0.05
    max_time = 60.
    seed = int(time.time())
    start_state = torch.tensor([-9, -9], **tensor_args)
    goal_state = torch.tensor([9, 8], **tensor_args)
    limits = torch.tensor([[-10, 10], [-10, 10]], **tensor_args)

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
        num_obst=15,
        rand_xy_limits=[[-7.5, 7.5], [-7.5, 7.5]],
        rand_rect_shape=[2, 2],
        tensor_args=tensor_args,
    )
    # For obst. generation
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    obst_map, obst_list = generate_obstacle_map(**obst_params)

    #-------------------------------- Planner ---------------------------------
    rrt_params = dict(
        n_dof=n_dof,
        opt_iters=opt_iters,
        start_state=start_state,
        limits=limits,
        cost=obst_map,
        step_size=step_size,
        n_radius=n_radius,
        max_time=max_time,
        goal_prob=goal_prob,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    planner = RRTStar(**rrt_params)

    #---------------------------------------------------------------------------
    # Optimize
    start = time.time()
    trajs = []
    for i in range(num_particle_per_goal):
        traj = planner.optimize(first_goal_return=True, debug=True, informed=True)
        planner.reset()
        if traj is not None:
            trajs.append(traj)
    print(f"{time.time() - start} seconds")
    
    #---------------------------------------------------------------------------
    # Plotting

    import numpy as np
    res = 200
    x = np.linspace(-10, 10, res)
    y = np.linspace(-10, 10, res)
    fig = plt.figure()
    # planner.render()
    ax = fig.gca()
    cs = ax.contourf(x, y, obst_map.map, 20, cmap='Greys')
    ax.plot(start_state[0], start_state[1], 'go', markersize=7)
    ax.plot(goal_state[0], goal_state[1], 'ro', markersize=7)
    ax.set_aspect('equal')
    for traj in trajs:
        ax.plot(traj[:, 0], traj[:, 1], 'b-', markersize=3)
    plt.show()
