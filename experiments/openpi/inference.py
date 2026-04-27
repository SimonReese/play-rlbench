import time
from typing import List

import numpy
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.tasks import SpatialTasks

import openpi_client.websocket_client_policy

from scipy.spatial.transform import Rotation


TASK = SpatialTasks.OBJECT_CONTAINER
IP = "127.0.0.1"
PORT = 4900


def main():
    remote_model = openpi_client.websocket_client_policy.WebsocketClientPolicy(IP, PORT)

    # Setup Environment
    cam_config = CameraConfig(image_size=(224, 224))
    obs_config = ObservationConfig(
        left_shoulder_camera= cam_config,
        right_shoulder_camera= cam_config,
        overhead_camera= cam_config,
        wrist_camera= cam_config,
        front_camera= cam_config,
        gripper_joint_positions= True
    )
    action_mode = MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=False, frame=RelativeFrame.EE), gripper_action_mode=Discrete())
    env = Environment(action_mode=action_mode, obs_config=obs_config)
    env.launch()
    # Open task
    task = env.get_task(TASK)
    task.set_variation(0)
    instr, obs = task.reset()
    # Let the simulation step a little
    for _ in range(10):
        obs, _, _ = task.step([0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 1.0, 0.99])
    
    print(instr)
    start_time = time.time()
    success = False
    while time.time() - start_time < 2000:
        front_img = numpy.ascontiguousarray(obs.front_rgb)
        wrist_img = numpy.ascontiguousarray(obs.wrist_rgb)
        gripper_open_amount = get_panda_gripper_open_amount(obs.gripper_joint_positions)[0]
        robot_state = numpy.concatenate((obs.joint_positions, [gripper_open_amount]), dtype=numpy.float32)
        # Pack obs
        packet = {
            "observation/image": front_img,
            "observation/wrist_image": wrist_img,
            "observation/state": robot_state,
            "instruction": instr[0]
        }
        action_chunk = remote_model.infer(packet)["actions"]
        
        for action in action_chunk:
            # temp fix for wrong output size
            action_padded = action[:-1]
            delta = action_padded[:3]
            rot_euler = action_padded[3:6]
            gripper_aperture = action_padded[-1]

            rot_quat = euler_to_quaternion(rot_euler)
            act = numpy.concatenate((delta, rot_quat, [gripper_aperture]))
            
            obs, reward, success = task.step(act)
        
        if success: break

        


# OTHER FUNCTIONS

def get_panda_gripper_open_amount(gripper_joint_positions: numpy.ndarray) -> List[float]:
    """Gets the gripper open state for the panda gripper. 1 means open, whilst 0 means closed.

    PANDA_JOINT_INTERVALS_LIST = [
        [0.0, 0.03999999910593033],
        [0.0, 0.03999999910593033]
    ]

    :param gripper_joint_positions: numpy.ndarray containing the current position of the gripper joints

    :return: A list of floats between 0 and 1 representing the gripper open
        state for each joint. 1 means open, whilst 0 means closed.
    """
    PANDA_JOINT_INTERVALS_LIST = [[0.0, 0.03999999910593033], [0.0, 0.03999999910593033]]
    joint_intervals_list = PANDA_JOINT_INTERVALS_LIST
    joint_intervals = numpy.array(joint_intervals_list)
    joint_range = joint_intervals[:, 1] - joint_intervals[:, 0]
    return list(numpy.clip((numpy.array(
        gripper_joint_positions) - joint_intervals[:, 0]) /
                        joint_range, 0.0, 1.0))

def euler_to_quaternion(euler: numpy.ndarray, format: str = "xyz") -> numpy.ndarray:
    rotation = Rotation.from_euler(format, euler)
    quaternion = rotation.as_quat()
    return quaternion

if __name__== "__main__" :
    main()