from car_env import CarEnv
import numpy as np
from util import Vec

env = CarEnv(file_name = 'maps/map3.txt', draw_rays=False)
env.color = (255,0,0)
while True:
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render(mode='trajectories')
