import numpy as np
import gym
import tensorflow as tf
from tqdm.auto import tqdm, trange
import pickle
from collections import defaultdict
import pathlib

from buffer import get_buffer

class Alg:
    def __init__(self, env, actor, critic, optimizer, alg, name = '1', buffer_size = 10**6, actions_per_epoch = 1,
            training_steps_per_epoch = 1, gamma = 0.99, polyak = 0.999, action_noise = (1, 0.1),
            train_summary_writer = None, test_summary_writer = None):
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env
        self.env_id = env.spec.id
        self.actor = actor
        self.critic = critic
        self.target_actor = tf.keras.models.clone_model(actor)
        self.target_critic = tf.keras.models.clone_model(critic)
        self.optimizer = optimizer
        self.alg = alg
        self.name = name
        self.buffer = get_buffer(buffer_size, self.env, name = name)

        self.actions_per_epoch = actions_per_epoch
        self.training_steps_per_epoch = training_steps_per_epoch
        self.gamma = gamma
        self.polyak = polyak
        self.train_summary_writer = train_summary_writer
        self.test_summary_writer = test_summary_writer
        self.stats = defaultdict(list) # switch to tensorboard
        self.current_episode_len = 0
        self.cumulative_reward = 0

        if callable(action_noise):
            self.action_noise = action_noise
        elif type(action_noise) is tuple: # interpolate linearly
            self.action_noise = lambda x: (1-x)*action_noise[0] + x*action_noise[1]
        else:
            self.action_noise = lambda x: action_noise
        

    def get_action(self, actor, state, noise=0):
        action = actor(np.array([state])).numpy()[0]
        if noise != 0:
            action = np.random.normal(loc=action, scale=noise) # action noise (change to parameter noise)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def train_step(self, batch_size):
        states, actions, rewards, next_states, done = self.buffer.sample(batch_size)
        self.train_step_critic(states, actions, rewards, next_states, done)
        self.train_step_actor(states)
        self.update_target_networks()

    def train_step_critic(self, states, actions, rewards, next_states, done):
        target_actions = self.target_actor(next_states)
        target_critic_input = np.hstack([next_states, target_actions])
        targets = rewards + self.gamma*(1-done)*tf.reshape(self.target_critic(target_critic_input), -1)    

        critic_input = np.hstack([states, actions])
        
        with tf.GradientTape() as tape:
            predictions = tf.reshape(self.critic(critic_input), -1)
            L = tf.reduce_mean(tf.square(targets - predictions), axis=0)

        gradients = tape.gradient(L, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

    def train_step_actor(self, states):
        with tf.GradientTape() as tape:
            actor_actions = self.actor(states)
            # negative loss so that optimizer maximizes instead of minimize
            Q = -tf.reduce_mean(self.critic(tf.concat([states, actor_actions], 1)), axis=0)
        gradients = tape.gradient(Q, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

    def update_target_networks(self):
        for cl, tcl in zip(self.critic.layers, self.target_critic.layers):
            tcl.set_weights(list(map(
                lambda tclw, clw: self.polyak*tclw + (1-self.polyak)*clw,
                tcl.get_weights(), cl.get_weights())))

        for al, tal in zip(self.actor.layers, self.target_actor.layers):
            tal.set_weights(list(map(
                lambda talw, alw: self.polyak*talw + (1-self.polyak)*alw,
                tal.get_weights(), al.get_weights())))

    def test(self, num_episodes = 1, render = True):
        fitness = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.actor(np.array([state])).numpy()[0]
                state, reward, done, _ = self.env.step(action)
                fitness += reward
        if self.test_summary_writer:
            with self.test_summary_writer.as_default():
                tf.summary.scalar('reward', fitness/num_episodes)
        self.stats['test_reward'].append(fitness/num_episodes)
        return fitness/num_episodes

    def save_models(self):
        directory = f'models/{self.alg}/{self.env_id}'
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        self.actor.save( f'{directory}/{self.name}-actor')
        self.critic.save(f'{directory}/{self.name}-critic')
