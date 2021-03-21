import numpy as np
import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from tqdm.auto import tqdm, trange
import pickle
from collections import defaultdict
from copy import deepcopy
import pathlib
from buffer import get_buffer


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

def to_numpy(var):
    return var.cpu().data.numpy()

class RLNN(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(RLNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = FloatTensor(max_action)

    def set_params(self, params):
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name),
                       map_location=lambda storage, loc: storage))

    def save_model(self, output, net_name):
        torch.save(
            self.state_dict(),
            '{}/{}.pkl'.format(output, net_name)
        )

class Actor(RLNN):
    def __init__(self, state_dim, action_dim, max_action, init=True):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 12)
        # self.l2 = nn.Linear(12, 12)
        self.l3 = nn.Linear(12, action_dim)


    def forward(self, x):
        x = torch.tanh(self.l1(x))
        # x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))*self.max_action

        return x

class Critic(RLNN):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 12)
        # self.l2 = nn.Linear(12, 12)
        self.l3 = nn.Linear(12, 1)

    def forward(self, state, action):
        x = torch.tanh(self.l1(torch.cat([state, action], 1)))
        # x = torch.tanh(self.l2(x))
        x = self.l3(x)

        return x

class ERL:
    def __init__(self, env, actor, critic, optimizer, doc = None, buffer_size = 10**6, fresh_buffer = False,
            actions_per_epoch = 1, training_steps_per_epoch = 1, gamma = 0.99, polyak = 0.995,
            action_noise = (1, 0.1), num_actors = 10, elite_frac = 0.2, episodes_per_actor = 1,
            episodes_rl_actor = 10, rl_actor_copy_every = 10, mutation_rate = 0.001, mutation_prob = 0.2,
            **kwargs):
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env
        self.env_id = env.spec.id
        self.actor = actor
        self.critic = critic
        self.target_actor = deepcopy(actor)
        self.target_critic = deepcopy(critic)
        # self.optimizer = optimizer
        self.alg = 'erl'
        self.buffer = get_buffer(buffer_size, self.env, doc = doc, fresh = fresh_buffer)

        self.actions_per_epoch = actions_per_epoch
        self.training_steps_per_epoch = training_steps_per_epoch
        self.gamma = gamma
        self.polyak = polyak
        # self.train_summary_writer = train_summary_writer
        # self.test_summary_writer = test_summary_writer
        self.stats = defaultdict(list) # switch to tensorboard
        self.current_episode_len = 0
        self.cumulative_reward = 0

        self.mse = nn.MSELoss()
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01)
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=0.01)

        # ERL things
        self.actors = [deepcopy(self.actor).cuda() for _ in range(num_actors)]
        self.elites = int(elite_frac*num_actors)
        self.episodes_per_actor = episodes_per_actor
        self.episodes_rl_actor = episodes_rl_actor
        self.rl_actor_copy_every = rl_actor_copy_every
        self.mutation_rate = mutation_rate
        self.mutation_prob = mutation_prob

        if callable(action_noise):
            self.action_noise = action_noise
        elif type(action_noise) is tuple: # interpolate linearly
            self.action_noise = lambda x: (1-x)*action_noise[0] + x*action_noise[1]
        else:
            self.action_noise = lambda x: action_noise

        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.target_actor = self.target_actor.cuda()
            self.critic = self.critic.cuda()
            self.target_critic = self.target_critic.cuda()

    def train(self, epochs, batch_size = 64, render = False, rl_actor_render = False,
              test_every = 100, render_test = True, render_mode = 'human', **kwargs):
        self.stats = defaultdict(list)

        self.env.color = (255, 255, 255)
        self.env.opacity = 20

        if not self.buffer.full():
            self.buffer.fill(self.env)
        
        print("Now training")
        with trange(epochs) as t:
            for epoch in t:
                # Evolutionary actors gather experience
                self.env.color = (0, 0, 255)
                self.env.opacity = 63
                rewards = []
                for actor in self.actors:
                    actor_reward = 0
                    for _ in range(self.episodes_per_actor):
                        state = self.env.reset()
                        done = False
                        while not done:
                            action = self.get_action(actor, state) # 0 noise
                            next_state, reward, done, _ = self.env.step(action)
                            self.buffer.store(state, action, reward, next_state, done)
                            state = next_state
                            actor_reward += reward
                            if render:
                                self.env.render(mode = render_mode)
                    rewards.append(actor_reward)
                self.stats['actors_rewards'].append(list(sorted(rewards)))
                rank = np.argsort(rewards)[::-1] # best on the front
                elites = [deepcopy(self.actors[i]) for i in rank[:self.elites]]
                rest = []
                for _ in range(len(self.actors)-self.elites): # tournament selection
                    a = np.random.choice(rank) # select from all
                    b = np.random.choice(rank)
                    if rewards[a] > rewards[b]: # maybe do crossover here
                        winner = self.actors[a]
                    else:
                        winner = self.actors[b]
                    winner = deepcopy(winner)
                    if random.random() < self.mutation_prob:
                        winner = self.mutate(winner, self.mutation_rate)
                    rest.append(winner)
                self.actors = elites+rest
                
                # RL actor gathers experience
                self.env.color = (255, 0, 0)
                self.env.opacity = 63
                total_reward = 0
                for _ in range(self.episodes_rl_actor):
                    state = self.env.reset()
                    done = False
                    while not done:
                        action = self.get_action(self.actor, state, self.action_noise(epoch/epochs))
                        next_state, reward, done, _ = self.env.step(action)
                        self.buffer.store(state, action, reward, next_state, done)
                        total_reward += reward
                        state = next_state
                        if render or rl_actor_render:
                            self.env.render(mode = render_mode)
                self.stats['reward'].append(total_reward/self.episodes_rl_actor)
                
                # Train RL actor based on experience
                for _ in range(self.training_steps_per_epoch):
                    self.train_step(batch_size)

                # Copy RL actor into population
                if (epoch+1) % self.rl_actor_copy_every == 0:
                    self.actors[-1] = deepcopy(self.actor)
                
                self.env.color = None
                if (epoch+1) % test_every == 0:
                    avg_fitness = self.test(1, render = render_test, render_mode = render_mode)
                    t.set_postfix(test_fitness = avg_fitness)
                    self.test(1, render = render_test, evo_actors = True, only_best = self.elites, render_mode = render_mode)

        return self.stats

    def get_action(self, actor, state, noise=0):
        action = to_numpy(actor(FloatTensor(state)))
        if noise != 0:
            action = np.random.normal(loc=action, scale=noise) # action noise (change to parameter noise)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def train_step(self, batch_size):
        states, actions, rewards, next_states, done = self.buffer.sample(batch_size)
        states = FloatTensor(states)
        actions = FloatTensor(actions)
        next_states = FloatTensor(next_states)
        rewards = FloatTensor(rewards)
        done = FloatTensor(done)
        self.train_step_critic(states, actions, rewards, next_states, done)
        self.train_step_actor(states)
        self.update_target_networks()

    def train_step_critic(self, states, actions, rewards, next_states, done):
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_actions = torch.clip(target_actions, self.env.action_space.low[0], self.env.action_space.high[0])
            reward_prediction = self.target_critic(next_states, target_actions)
            targets = rewards + self.gamma*(1-done)*torch.reshape(reward_prediction, (-1,))
        
        critic_ratings = torch.reshape(self.critic(states, actions), (-1,))
        L = self.mse(critic_ratings, targets) # MSE loss

        self.stats['critic_loss'].append(to_numpy(L))
        self.stats['critic_pred'].append(to_numpy(critic_ratings))
        self.stats['critic_weights'].append(list(self.critic.parameters()))

        self.critic_optimizer.zero_grad()
        L.backward()
        self.critic_optimizer.step()

    def train_step_actor(self, states):
        actor_actions = self.actor(states)
        # negative loss so that optimizer maximizes (the critic's rating) instead of minimize
        Q = -self.critic(states, actor_actions).mean()

        self.stats['actor_loss'].append(to_numpy(Q))
        self.stats['actor_pred'].append(to_numpy(actor_actions))
        self.stats['actor_weights'].append(list(self.actor.parameters()))

        self.actor_optimizer.zero_grad()
        Q.backward()
        self.actor_optimizer.step()

    def update_target_networks(self):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                (1 - self.polyak) * param.data + self.polyak * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(
                (1 - self.polyak) * param.data + self.polyak * target_param.data)

    def mutate(self, actor, noise):
        actor = actor.cpu()
        for param in actor.parameters():
            param.data.copy_(FloatTensor(np.random.normal(loc=param.data, scale=noise)))
        actor = actor.cuda()
        return actor

    def test(self, num_episodes = 1, render = True, evo_actors = False, only_best = None, render_mode = 'human'):
        fitness = 0
        if not evo_actors:
            # Testing RLActor
            for _ in range(num_episodes):
                state = self.env.reset()
                done = False
                while not done:
                    action = to_numpy(self.actor(FloatTensor(state)))
                    state, reward, done, _ = self.env.step(action)
                    if render or (render_mode == 'trajectories' and done):
                        self.env.color = None
                        self.env.render(mode = render_mode)
                    fitness += reward
            self.stats['test_reward'].append(fitness/num_episodes)
            return fitness/num_episodes
        else:
            # Testing EVOActors
            if only_best is None:
                only_best = len(self.actors)
            actors = self.actors[:only_best]
            for actor in actors:
                state = self.env.reset()
                done = False
                while not done:
                    action = to_numpy(actor(FloatTensor(state)))
                    state, reward, done, _ = self.env.step(action)
                    if render or (render_mode == 'trajectories' and done):
                        self.env.color = None
                        self.env.render(mode = render_mode)
                    fitness += reward
            return fitness/len(actors)
