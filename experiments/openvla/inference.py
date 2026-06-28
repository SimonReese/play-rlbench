import os
import time
from typing import List, Union

import imageio
import numpy
import rlbench.utils
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig

from scipy.spatial.transform import Rotation
from pathlib import Path

import requests
import json_numpy
json_numpy.patch()

# Environment
HEADLESS = True

IP = "titan2.dei.unipd.it"
PORT = 8000

DATASET_PATH = "/home/peraro/source/play-rlbench/data-generation/datasets/generated-20ep"

SAVE_STATS = True
SAVE_VIDEO = True
STATS_PATH =  Path("/home/peraro/source/play-rlbench/experiments/results/openvla", Path(DATASET_PATH).name)  
VIDEO_PATH = Path("/home/peraro/source/play-rlbench/experiments/results/openvla", Path(DATASET_PATH).name)

def exec_task(env: Environment, task_name: str, variation: Union[int, None] = None, demo: Union[Demo, None] = None, demo_idx = None,
              save_stats = True, save_video = False, stats_path = None, video_path = None,
              stat_filename = "stats.txt"
              ):
    
    # Manage stats paths
    if save_stats:
        if stats_path == None: # Default stats path
            stats_path = Path("./", stat_filename)
        else:
            stats_path = Path(stats_path)
            stats_path.mkdir(parents=True, exist_ok=True)
            stats_path.joinpath(stat_filename)
    
    # Manage video path
    if save_video:
        if video_path == None:
            video_path = Path("./")
        else:
            video_path = Path(video_path, task_name, "variation" + str(variation), "episode" + str(demo_idx))
            video_path.mkdir(parents=True, exist_ok=True)


    # Open task
    task = rlbench.utils.name_to_task_class(task_name)
    task = env.get_task(task)
    task.set_variation(variation if variation is not None else 0)
    instr, obs = task.reset_to_demo(demo)
    # Let the simulation step a little
    for _ in range(10):
        obs, _, _ = task.step()
    
    print(instr)
    start_time = time.time()
    success = False
    front_camera_images = []
    while time.time() - start_time < 180:
        front_img = numpy.ascontiguousarray(obs.front_rgb)        
        # Pack obs
        packet = {
            "image": front_img,
            "instruction": instr[0]
        }
        action = requests.post(
            url=f"http://{IP}:{PORT}/act",
            json=packet
        ).json()
        
        
        front_camera_images.append(obs.front_rgb) 
        delta = action[:3]
        rot_euler = action[3:6]
        gripper_aperture = action[-1]
        #print(f"Action :{delta}, {rot_euler}, {gripper_aperture}")
        if gripper_aperture < 0.8: gripper_aperture = 0.0
        rot_quat = rlbench.utils.euler_to_quaternion(rot_euler)
        act = numpy.concatenate((delta, rot_quat, [gripper_aperture]))
            
        try:
            obs, reward, success = task.step(act)
        except InvalidActionError:
            print("Cannot reach")
            obs = task.get_observation()
            task.step()
        if success: break

    # write stats
    if save_stats:
        with open(stats_path._str + "/" + stat_filename, "a") as stats_file:
            stats_file.write(f"{task_name}:variation{variation}:episode{demo_idx}:{success} {instr[0]}\n")
            stats_file.flush()
    
    if save_video:
        with imageio.get_writer(video_path._str + "/front_cam.mp4") as vid:
            for frame in front_camera_images:
                vid.append_data(frame)

def main():
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
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=HEADLESS)
    env.launch()
    
    task_list = os.listdir(DATASET_PATH)
    for task_name in task_list:
        print(f"Opening task {task_name}")
        
        variations_ids = rlbench.utils.get_variations_ids(DATASET_PATH, task_name)
        # Open each variation
        for variation in variations_ids:
            print("Opening variation ", variation)
            # Get all episodes in variation
            episodes_demos = rlbench.utils.get_stored_demos(amount=-1, image_paths=True, dataset_root=DATASET_PATH, 
                                        variation_number=variation, task_name=task_name,obs_config=obs_config, random_selection=False)
            
            for idx, demo in enumerate(episodes_demos):
                exec_task(env, task_name, variation, demo, idx,
                          SAVE_STATS, SAVE_VIDEO, stats_path=str(STATS_PATH), video_path=str(VIDEO_PATH))
                

if __name__== "__main__" :
    main()