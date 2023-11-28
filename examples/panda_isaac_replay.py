import os
from pprint import pprint

from future.moves import pickle

from torch_robotics.isaac_gym_envs.motion_planning_envs import MotionPlanningIsaacGymEnv, MotionPlanningController

from torch_robotics.environments.grasped_objects import GraspedObjectBox

import einops
import torch

from torch_robotics.environments.env_spheres_3d import EnvSpheres3D
from torch_robotics.environments.env_spheres_3d_extra_objects import EnvSpheres3DExtraObjects
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
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

# ---------------------------- Environment, Robot, PlanningTask ---------------------------------
env = EnvSpheres3D(
    tensor_args=tensor_args
)

robot = RobotPanda(
    use_self_collision_storm=False,
    grasped_object=GraspedObjectBox(
        attached_to_frame=RobotPanda.link_name_ee,
        object_collision_margin=0.05,
        tensor_args=tensor_args
    ),
    gripper=True,
    tensor_args=tensor_args
)

task = PlanningTask(
    env=env,
    robot=robot,
    ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),  # workspace limits
    margin_for_waypoint_collision_checking=0.01,
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

motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
    env, robot, task,
    asset_root=get_robot_path().as_posix(),
    robot_asset_file=robot.urdf_robot_file.replace(get_robot_path().as_posix() + '/', ''),
    num_envs=trajs_pos.shape[1],
    all_robots_in_one_env=True,

    show_viewer=True,
    sync_viewer_with_real_time=False,
    viewer_time_between_steps=0.1,

    render_camera_global=True,

    color_robots=False,
    draw_goal_configuration=True,
    draw_collision_spheres=False,
    draw_contact_forces=False,
    draw_end_effector_frame=True,
    draw_end_effector_path=False,

    camera_global_from_top=True if env.dim == 2 else False,
    add_ground_plane=False if env.dim == 2 else True,
)

motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
isaac_statistics = motion_planning_controller.run_trajectories(
    trajs_pos,
    start_states_joint_pos=trajs_pos[0], goal_state_joint_pos=trajs_pos[-1][0],
    n_first_steps=n_first_steps, n_last_steps=n_last_steps,
    stop_robot_if_in_contact=False,
    make_video=True, video_duration=5.,
    video_path=base_file_name + 'isaac-controller-position.mp4',
    make_gif=False
)
print('-----------------')
print(f'isaac_statistics:')
pprint(isaac_statistics)
print('-----------------')



