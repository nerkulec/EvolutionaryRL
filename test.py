from car_env import CarEnv
import numpy as np
from util import Vec

env = CarEnv(file_name = 'maps/maptest.txt')

while True:
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
    env.render()
