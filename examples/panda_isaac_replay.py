import os

from future.moves import pickle

from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

from torch_robotics.environments.objects import GraspedObjectPandaBox

import einops
import torch

from torch_robotics.environments.env_spheres_3d import EnvSpheres3D
from torch_robotics.environments.env_spheres_3d_extra_objects import EnvSpheres3DExtraObjects
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.trajectory.utils import interpolate_traj_via_points

seed = 0
fix_random_seed(seed)

device = get_torch_device()
tensor_args = {'device': 'cpu', 'dtype': torch.float32}

# ---------------------------- Load motion planning results ---------------------------------
# base_file_name = 'panda_spheres_GPMP'
base_file_name = 'panda_spheres_CHOMP'
with open(os.path.join('./', f'{base_file_name}-results_data_dict.pickle'), 'rb') as handle:
    results_planning = pickle.load(handle)


base_file_name = '/home/carvalho/Projects/MotionPlanningDiffusion/mpd/data_trajectories/EnvSpheres3D-RobotPanda/2'
# base_file_name = '/home/carvalho/Projects/MotionPlanningDiffusion/mpd/data_trajectories/EnvSpheres3D-RobotPanda/0'
# base_file_name = '/home/carvalho/Projects/MotionPlanningDiffusion/mpd/scripts/generate_data/logs/generate_trajectories_2023-10-19_19-13-15/env_id___EnvSpheres3D/robot_id___RobotPanda/16'
base_file_name = '/home/carvalho/Projects/MotionPlanningDiffusion/mpd/data_trajectories/EnvSpheres3D-RobotPanda/200'

with open(os.path.join(base_file_name, f'results_data_dict.pickle'), 'rb') as handle:
    results_planning = pickle.load(handle)

# ---------------------------- Environment, Robot, PlanningTask ---------------------------------
env = EnvSpheres3D(
    tensor_args=tensor_args
)

robot = RobotPanda(
    # grasped_object=GraspedObjectPandaBox(tensor_args=tensor_args),
    use_self_collision_storm=True,
    use_collision_spheres=True,
    tensor_args=tensor_args
)

task = PlanningTask(
    env=env,
    robot=robot,
    ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),  # workspace limits
    tensor_args=tensor_args
)

# -------------------------------- Physics ---------------------------------
trajs_iters = results_planning['trajs_iters_free']
trajs_pos = robot.get_position(trajs_iters[-1]).movedim(1, 0)
trajs_vel = robot.get_velocity(trajs_iters[-1]).movedim(1, 0)

# POSITION CONTROL
# add initial positions for better visualization
n_first_steps = 30
n_last_steps = 30
# trajs_pos = interpolate_traj_via_points(trajs_pos.movedim(0, 1), 2).movedim(1, 0)
# trajs_pos = trajs_pos[:, 0, :].unsqueeze(1)

results_planning['dt'] = 1./10.

motion_planning_isaac_env = PandaMotionPlanningIsaacGymEnv(
    env, robot, task,
    asset_root="/home/carvalho/Projects/MotionPlanningDiffusion/mpd/deps/isaacgym/assets",
    controller_type='position',
    num_envs=trajs_pos.shape[1],
    all_robots_in_one_env=True,
    color_robots=False,
    show_goal_configuration=True,
    sync_with_real_time=True,
    show_collision_spheres=False,
    **results_planning,
    # show_collision_spheres=True
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
    video_path=os.path.join(base_file_name, 'isaac-controller-position.mp4'),
    make_gif=False
)

