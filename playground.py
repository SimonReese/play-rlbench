import time
from typing import Literal

from RLBench.rlbench.action_modes.action_mode import IdleActionMode
from RLBench.rlbench.environment import Environment
from RLBench.rlbench.tasks import PushButton
from RLBench.rlbench.tasks import SPATIAL_TASKS, SpatialTasks

import imageio

def main(): 
    

    act_mode = IdleActionMode()

    # Start env
    env = Environment(act_mode, headless=True)
    env.launch()

    # Load task
    task = env.get_task(SpatialTasks.SLIDE_BLOCK)
    task.reset()
    
    # Act
    start = time.time()
    frames = []
    demos = task.get_demos(1, True)
    demo = demos.pop()
    for obs in demo:
        frames.append(obs.front_rgb)
        print(demo.demo_description)
    # while time.time() - start < 5:
    #     obs, reward, done = task.step([])
    #     frames.append(obs.front_rgb)
    imageio.mimsave('output.mp4', frames, fps=30)

    

if __name__=="__main__":
    main()