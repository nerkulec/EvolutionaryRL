import numpy as np
import gym
import tensorflow as tf
from tqdm.auto import tqdm, trange
import pickle
from collections import defaultdict
import pathlib

from buffer import get_buffer
from alg import Alg

class DDPG(Alg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 'ddpg', **kwargs)

    def train(self, epochs, batch_size = 64, render = False, test_every = 100, render_test = True):
        self.stats = defaultdict(list)

        state = self.env.reset()
        
        if not self.buffer.full():
            self.buffer.fill(self.env)
        
        print("Now training")
        with trange(epochs) as t:
            for epoch in t:
                for _ in range(self.actions_per_epoch):
                    action = self.get_action(self.actor, state, self.action_noise(epoch/epochs))
                    state = self.step(state, action)
                    if render:
                        self.env.render()
                if len(self.stats['reward']) > 0:
                    t.set_postfix(train_fitness = self.stats['reward'][-1])
                for _ in range(self.training_steps_per_epoch):
                    self.train_step(batch_size)
                if (epoch+1) % test_every == 0:
                    avg_fitness = self.test(1, render = render_test)
                    t.set_postfix(test_fitness = avg_fitness)

        return self.stats

    def step(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        self.buffer.store(state, action, reward, next_state, done)
        self.current_episode_len += 1
        self.cumulative_reward += reward
        if done:
            if self.train_summary_writer:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('reward', self.current_episode_len)
            self.stats['reward'].append(self.cumulative_reward)
            self.cumulative_reward = 0
            self.current_episode_len = 0
            return self.env.reset()
        else:
            return next_state
