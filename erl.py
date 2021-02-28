import numpy as np
import random
import gym
import tensorflow as tf
from tqdm.auto import tqdm, trange
import pickle
from collections import defaultdict
import pathlib

from buffer import get_buffer
from alg import Alg

clone_without_weights = tf.keras.models.clone_model
def clone_with_weights(model):
    new_model = clone_without_weights(model)
    new_model.set_weights(model.get_weights())
    return new_model

class ERL(Alg):
    def __init__(self, *args, num_actors = 10, elite_frac = 0.2, episodes_per_actor = 1, episodes_rl_actor = 10,
                 rl_actor_copy_every = 10, mutation_rate = 0.001, mutation_prob = 0.2, **kwargs):
        super().__init__(*args, 'erl', **kwargs)
        self.actors = [clone_without_weights(self.actor) for _ in range(num_actors)]
        self.elites = int(elite_frac*num_actors)
        self.episodes_per_actor = episodes_per_actor
        self.episodes_rl_actor = episodes_rl_actor
        self.rl_actor_copy_every = rl_actor_copy_every
        self.mutation_rate = mutation_rate
        self.mutation_prob = mutation_prob

    def train(self, epochs, batch_size = 64, render = False, rl_actor_render = False,
              test_every = 100, render_test = True):
        self.stats = defaultdict(list)

        if not self.buffer.full():
            self.buffer.fill(self.env)
        
        print("Now training")
        with trange(epochs) as t:
            for epoch in t:
                # Evolutionary actors gather experience
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
                            # actor_reward += reward
                            if render:
                                self.env.render()
                        actor_reward += reward
                    rewards.append(actor_reward)
                self.stats['actors_rewards'].append(list(sorted(rewards)))
                rank = np.argsort(rewards)[::-1] # best on the front
                elites = [clone_with_weights(self.actors[i]) for i in rank[:self.elites]]
                rest = []
                for _ in range(len(self.actors)-self.elites): # tournament selection
                    # a = np.random.choice(rank[self.elites:]) # maybe select from all
                    # b = np.random.choice(rank[self.elites:])
                    a = np.random.choice(rank) # select from all
                    b = np.random.choice(rank)
                    if rewards[a] > rewards[b]: # maybe do crossover here
                        winner = self.actors[a]
                    else:
                        winner = self.actors[b]
                    winner = clone_with_weights(winner)
                    if random.random() < self.mutation_prob:
                        winner = self.mutate(winner, self.mutation_rate)
                    rest.append(winner)
                self.actors = elites+rest
                
                # RL actor gathers experience
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
                            self.env.render()
                self.stats['reward'].append(total_reward/self.episodes_rl_actor)
                
                # Train RL actor based on experience
                for _ in range(self.training_steps_per_epoch):
                    self.train_step(batch_size)

                # Copy RL actor into population
                if (epoch+1) % self.rl_actor_copy_every == 0:
                    self.actors[-1] = clone_with_weights(self.actor)

                if (epoch+1) % test_every == 0:
                    avg_fitness = self.test(1, render = render_test)
                    t.set_postfix(test_fitness = avg_fitness)
                    self.test(1, render = render_test, rl_actors = True, only_best = self.elites)

        return self.stats

    def mutate(self, actor, noise):
        for actor_layer in actor.layers:
            actor_layer.set_weights(list(map(
                lambda w: np.random.normal(loc=w, scale=noise), # parameter noise
                actor_layer.get_weights())))
        return actor


    def test(self, num_episodes = 1, render = True, rl_actors = False, only_best = None):
        fitness = 0
        if not rl_actors:
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
        else:
            if only_best is None:
                only_best = len(self.actors)
            actors = self.actors[:only_best]
            for actor in actors:
                state = self.env.reset()
                done = False
                while not done:
                    if render:
                        self.env.render()
                    action = actor(np.array([state])).numpy()[0]
                    state, reward, done, _ = self.env.step(action)
                    fitness += reward
            return fitness/len(actors)