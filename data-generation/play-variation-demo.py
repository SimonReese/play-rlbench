"""
This file s consider a dir structure like peract dataset to open each specific variationID a for each task and play a specific episode/demo
There can still be differences in things like colours
"""

from typing import List, Union
import pickle

import os
import numpy
from scipy.spatial.transform import Rotation
from rlbench.action_modes.action_mode import MoveArmThenGripper, ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.backend.const import LOW_DIM_PICKLE
from rlbench.demo import Demo

import rlbench.utils

# Load all demos for variation
def load_demos(dataset_path: str, task_name:str, variation_id:int , VARIATION_FOLDER_PREFIX = "variation") -> List[Demo]:
    """Load all episodes for a given variation"""
    demos = []
    # Open variation folder
    variation_folder = os.path.join(dataset_path, task_name, f"{VARIATION_FOLDER_PREFIX}{variation_id}", "episodes")
    episodes = os.listdir(os.path.join(variation_folder))
    for episode in episodes:
        episode_path = os.path.join(variation_folder, episode)
        with open(os.path.join(episode_path, LOW_DIM_PICKLE), "rb") as pfile:
            demo: Demo = pickle.load(pfile)
            varfile = open(os.path.join(episode_path, "variation_number.pkl"), "rb")
            index = pickle.load(varfile)
            demo._observations[0].misc["variation_index"] = index
            demos.append(demo)

    return demos

def invert_quaternion(quaternion: numpy.ndarray, format = "xyzw") -> numpy.ndarray:
    if format == "xyzw":
        return numpy.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]) / numpy.dot(quaternion, quaternion)
    else: #wxyz
        return numpy.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]) / numpy.dot(quaternion, quaternion)

def quaternion_multiplication(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return numpy.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


# ---------QUATERNIONS UTILITY-----------
def pose_to_T(p, q):
    T = numpy.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = p
    return T

def invert_T(T):
    Rm = T[:3, :3]
    p = T[:3, 3]

    T_inv = numpy.eye(4)
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ p
    return T_inv

def delta_pose_ee(p_cur, q_cur, p_des, q_des):
    """
    Returns delta pose in EE frame as:
    [dx, dy, dz, qx, qy, qz, qw]
    """
    T_cur = pose_to_T(p_cur, q_cur)
    T_des = pose_to_T(p_des, q_des)

    T_delta = invert_T(T_cur) @ T_des

    p_delta = T_delta[:3, 3]
    q_delta = Rotation.from_matrix(T_delta[:3, :3]).as_quat()

    return numpy.hstack((p_delta, q_delta))

def quaternion_to_euler(quaternion: numpy.ndarray) -> numpy.ndarray:
    rotation = Rotation.from_quat(quaternion)
    angles = rotation.as_euler("xyz")
    return angles

def euler_to_quaternion(euler: numpy.ndarray, format: str = "xyz") -> numpy.ndarray:
    rotation = Rotation.from_euler(format, euler)
    quaternion = rotation.as_quat()
    return quaternion

# ---------OTHER UTILITY---------------
def pretty_print_array(data: numpy.ndarray, printout=False, label="") -> Union[numpy.ndarray , None]:
    pretty = data.clip(min=1e-3)
    if printout:
        print(f"{label}{pretty}")
        return
    return pretty

def get_open_amount(gripper_joint_positions: numpy.ndarray) -> List[float]:
        """Gets the gripper open state. 1 means open, whilst 0 means closed.

        :return: A list of floats between 0 and 1 representing the gripper open
            state for each joint. 1 means open, whilst 0 means closed.
        """
        _, joint_intervals_list = ([False, False], [[0.0, 0.03999999910593033], [0.0, 0.03999999910593033]]) #self.get_joint_intervals()
        joint_intervals = numpy.array(joint_intervals_list)
        joint_range = joint_intervals[:, 1] - joint_intervals[:, 0]
        return list(numpy.clip((numpy.array(
            gripper_joint_positions) - joint_intervals[:, 0]) /
                            joint_range, 0.0, 1.0))
# ----------------------


class IdleAction(ActionMode):

    def action(self, scene, action):
        scene.step()

#act = IdleAction(ArmActionMode(), Discrete())
cam_config = CameraConfig(image_size=(224, 224))
obs_config = ObservationConfig(
    left_shoulder_camera= cam_config,
    right_shoulder_camera= cam_config,
    overhead_camera= cam_config,
    wrist_camera= cam_config,
    front_camera= cam_config,
    gripper_joint_positions=True
)
arm_action_mode = EndEffectorPoseViaPlanning(absolute_mode=False, frame=RelativeFrame.EE)
act = MoveArmThenGripper(arm_action_mode=arm_action_mode, gripper_action_mode=Discrete())
env = Environment(action_mode=act)
env.launch()
print(f"Shape {env.action_shape}")


# Wait till popup shows
"""for i in range(100):
    assert env._scene is not None and env._scene.pyrep is not None
    env._scene.pyrep.step()"""

dataset  = "/home/peraro/source/play-rlbench/data-generation/datasets/generated-16-04-00-00"
tasks_list = os.listdir(dataset)

# Open all tasks envs
for task_name in tasks_list:
    print(f"Opening task {task_name}")
    task_class = rlbench.utils.name_to_task_class(task_name)
    task = env.get_task(task_class)

    variations_ids = rlbench.utils.get_variations_ids(dataset, task_name)
    # Open each variation
    for variation in variations_ids:
        print("Opening variation ", variation)
        # Get all episodes in variation
        episodes = rlbench.utils.get_stored_demos(amount=-1, image_paths=True, dataset_root=dataset, 
                                       variation_number=variation, task_name=task_name,obs_config=obs_config, random_selection=False)
        task.set_variation(variation)
        #demos = load_demos(dataset, task_name, variation)

        for num, demo in enumerate(episodes):
            print(f"Opening episode {num}")
            description , _local_obs = task.reset_to_demo(demo)
            print(f"Description {description}")
            old_pose = _local_obs.gripper_pose
            # Reproduce each step
            for obs in demo:
                new_pose = obs.gripper_pose
                delta_action = delta_pose_ee(old_pose[:3], old_pose[3:], new_pose[:3], new_pose[3:])
                print(f"Gripper joint positions {pretty_print_array(obs.gripper_joint_positions)}, is open: {obs.gripper_open}")
                print(f"Open amount{get_open_amount(obs.gripper_joint_positions)[0]} {obs.gripper_open=}")
                print(f"Joint Intervals{task._scene.robot.gripper.get_joint_intervals()}")
                #print(f"Start: {old_pose}\nTarget: {new_pose}\nDelta: {delta_action}")
                #print(f"Rotation will be {pretty_print_array(quaternion_to_euler(delta_action[3:]))}")
                
                _local_obs, reward, done = task.step(numpy.concatenate((delta_action, [obs.gripper_open])))
                old_pose = _local_obs.gripper_pose
            print(f"Task done")
        exit()
            

#env.shutdown()
