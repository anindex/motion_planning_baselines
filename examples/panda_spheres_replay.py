import os

from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import einops
import torch

from torch_robotics.environment.env_spheres_3d import EnvSpheres3D
from torch_robotics.robot.robot_panda import RobotPanda
from torch_robotics.task.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.trajectory.utils import interpolate_traj_via_points

seed = 0
fix_random_seed(seed)

device = get_torch_device()
tensor_args = {'device': 'cpu', 'dtype': torch.float32}

# ---------------------------- Environment, Robot, PlanningTask ---------------------------------
env = EnvSpheres3D(
    precompute_sdf_obj_fixed=True,
    sdf_cell_size=0.01,
    tensor_args=tensor_args
)

robot = RobotPanda(tensor_args=tensor_args)

task = PlanningTask(
    env=env,
    robot=robot,
    ws_limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # workspace limits
    tensor_args=tensor_args
)

# -------------------------------- Physics ---------------------------------

# trajs_iters = torch.load('trajs_iters.pt')
traj_iters_path = 'trajs_iters_panda_spheres_GPMP.pt'
# traj_iters_path = 'trajs_iters_panda_spheres_HybridPlanner.pt'

traj_iters_base = os.path.splitext(traj_iters_path)[0]

trajs_iters = torch.load(traj_iters_path)


trajs_pos = robot.get_position(trajs_iters[-1]).movedim(1, 0)
trajs_vel = robot.get_velocity(trajs_iters[-1]).movedim(1, 0)

# POSITION CONTROL
# add initial positions for better visualization
n_first_steps = 100
n_last_steps = 100
trajs_pos = interpolate_traj_via_points(trajs_pos.movedim(0, 1), 2).movedim(1, 0)

motion_planning_isaac_env = PandaMotionPlanningIsaacGymEnv(
    env, robot, task,
    asset_root="/home/carvalho/Projects/MotionPlanningDiffusion/mpd/isaacgym/assets",
    controller_type='position',
    num_envs=trajs_pos.shape[1],
    all_robots_in_one_env=True,
    color_robots=False,
    show_goal_configuration=True,
    sync_with_real_time=True
)

motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
motion_planning_controller.run_trajectories(
    trajs_pos,
    start_states_joint_pos=trajs_pos[0], goal_state_joint_pos=trajs_pos[-1][0],
    n_first_steps=n_first_steps,
    n_last_steps=n_last_steps,
    visualize=True,
    render_viewer_camera=True,
    make_video=True,
    video_path=f'{traj_iters_base}-controller-position.mp4',
    make_gif=False
)




# VELOCITY CONTROL
# add initial and final velocities multiple times
trajs_vel = interpolate_traj_via_points(trajs_vel.movedim(0, 1), 1).movedim(1, 0)

motion_planning_isaac_env = PandaMotionPlanningIsaacGymEnv(
    env, robot, task,
    asset_root="/home/carvalho/Projects/MotionPlanningDiffusion/mpd/isaacgym/assets",
    controller_type='velocity',
    num_envs=trajs_pos.shape[1],
    all_robots_in_one_env=True,
    color_robots=False,
    show_goal_configuration=True,
    sync_with_real_time=True
)

motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
motion_planning_controller.run_trajectories(
    trajs_vel,
    start_states_joint_pos=trajs_pos[0], goal_state_joint_pos=trajs_pos[-1][0],
    n_first_steps=n_first_steps,
    n_last_steps=n_last_steps,
    visualize=True,
    render_viewer_camera=True,
    make_video=True,
    video_path=f'{traj_iters_base}-controller-velocity.mp4',
    make_gif=False
)
