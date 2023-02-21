import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar
from mp_baselines.planners.utils import smoothen_trajectory
from robot_envs.base_envs.planar_2_link_robot_obstacle_map_env import RobotPlanarTwoLink

if __name__ == "__main__":
    seed = 2
    fix_random_seed(seed)

    tensor_args = {'device': 'cpu', 'dtype': torch.float64}

    # -------------------------------- Environment ---------------------------------
    env = RobotPlanarTwoLink(tensor_args=tensor_args)

    # -------------------------------- Planner ---------------------------------
    n_iters = 30000
    max_best_cost_iters = 5000
    cost_eps = 1e-2
    step_size = np.pi/50
    n_radius = np.pi/4
    n_knn = 5
    goal_prob = 0.1
    max_time = 60.

    start_state = torch.tensor([-np.pi/2, 0], **tensor_args)
    goal_state = torch.tensor([np.pi-0.05, 0], **tensor_args)

    rrt_params = dict(
        env=env,
        n_iters=n_iters,
        start_state=start_state,
        step_size=step_size,
        n_radius=n_radius,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    planner = RRTConnect(**rrt_params)

    # ---------------------------------------------------------------------------
    # Optimize
    start = time.time()
    traj_raw = planner.optimize(debug=True, informed=True)
    print(f"{time.time() - start} seconds")

    # ---------------------------------------------------------------------------
    # Plotting
    traj_pos, traj_vel = None, None
    if traj_raw is not None:
        traj_pos, traj_vel = smoothen_trajectory(traj_raw, traj_len=256, tensor_args=tensor_args)

        fig, ax = plt.subplots()

        def animate_fn(i):
            ax.clear()

            env.obstacle_map.plot(ax)

            q = traj_pos[i]
            q = q.view(1, 1, -1)
            env.render(q, ax)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')


        animation_time_in_seconds = 5
        print('Creating animation...')
        ani = FuncAnimation(fig, animate_fn,
                            frames=len(traj_pos),
                            interval=animation_time_in_seconds * 1000 / len(traj_pos),
                            repeat=False)
        print('...finished Creating animation')

        print('Saving video...')
        ani.save(os.path.join('planar_2_link_rrt_connect.mp4'), fps=int(len(traj_pos) / animation_time_in_seconds), dpi=90)
        print('...finished Saving video')

    # ----------------- Space of collision-free configurations -----------------
    qs_free = planner.create_uniform_samples(10000, max_samples=10000)
    fig, ax1 = plt.subplots()
    planner.render(ax1)
    ax1.scatter(qs_free[:, 0], qs_free[:, 1], color='grey', alpha=0.2, s=5)
    ax1.scatter(start_state[0], start_state[1], color='green', marker='o', zorder=10, s=50)
    ax1.scatter(goal_state[0], goal_state[1], color='red', marker='o', zorder=10, s=50)
    if traj_pos is not None:
        ax1.plot(traj_pos[:, 0], traj_pos[:, 1], color='red', linewidth=2.)
    if traj_vel is not None:
        ax1.quiver(traj_pos[:, 0], traj_pos[:, 1], traj_vel[:, 0], traj_vel[:, 1], color='blue', linewidth=2., zorder=11)
    ax1.set_xlim(env.q_min[0], env.q_max[0])
    ax1.set_ylim(env.q_min[1], env.q_max[1])
    ax1.set_aspect('equal')

    # ----------------- Position and velocity trajectories -----------------
    if traj_pos is not None and traj_vel is not None:
        fig, ax2 = plt.subplots(2, 1, squeeze=False)
        for d in range(traj_pos.shape[-1]):
            ax2[0, 0].plot(np.arange(traj_pos.shape[0]), traj_pos[:, d], label=f'{d}')
            ax2[1, 0].plot(np.arange(traj_vel.shape[0]), traj_vel[:, d], label=f'{d}')
        ax2[0, 0].set_title('Position')
        ax2[1, 0].set_title('Velocity')

    plt.legend()
    plt.show()
