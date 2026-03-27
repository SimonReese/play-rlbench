from RLBench.rlbench.action_modes.action_mode import IdleActionMode
from RLBench.rlbench.environment import Environment
from RLBench.rlbench.tasks import PushButton


def main():

    act_mode = IdleActionMode()

    # Start env
    env = Environment(act_mode)
    env.launch()

    # Load task
    task = env.get_task(PushButton)
    task.reset()
    
    # Act
    while True:
        obs, reward, done = task.step([])

    pass

if __name__=="__main__":
    main()