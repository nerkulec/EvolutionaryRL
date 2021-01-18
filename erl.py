import numpy as np
import gym
import tensorflow as tf
from tqdm.auto import tqdm, trange
import pickle
from collections import defaultdict
import pathlib

from buffer import get_buffer
from alg import Alg

clone = tf.keras.models.clone_model

class ERL(Alg):
    def __init__(self, *args, num_actors = 10, elite_frac = 0.2, episodes_per_actor = 5, episodes_rl_actor = 10,
                 rl_actor_copy_every = 10, mutation_rate = 0.001, **kwargs):
        super().__init__(*args, 'erl', **kwargs)
        self.actors = [clone(self.actor) for _ in range(num_actors)]
        self.elites = int(elite_frac*num_actors)
        self.episodes_per_actor = episodes_per_actor
        self.episodes_rl_actor = episodes_rl_actor
        self.rl_actor_copy_every = rl_actor_copy_every
        self.mutation_rate = mutation_rate

    def train(self, epochs, batch_size = 64, render = False, rl_actor_render = False,
              test_every = 100, render_test = True):
        self.stats = defaultdict(list)

        if not self.buffer.full():
            self.buffer.fill()
        
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
                            actor_reward += reward
                            if render:
                                self.env.render()
                    rewards.append(actor_reward)
                self.stats['actors_rewards'].append(list(sorted(rewards)))
                rank = np.argsort(rewards)
                elites = [clone(self.actors[i]) for i in rank[-self.elites:]]
                rest = []
                for _ in range(len(self.actors)-self.elites):
                    a = np.random.choice(rank[:self.elites])
                    b = np.random.choice(rank[:self.elites])
                    if rewards[a] > rewards[b]: # maybe do crossover here
                        rest.append(self.mutate(clone(self.actors[a]), self.mutation_rate))
                    else:
                        rest.append(self.mutate(clone(self.actors[b]), self.mutation_rate))
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
                    self.actors[-1] = clone(self.actor)

                if (epoch+1) % test_every == 0:
                    avg_fitness = self.test(5, render = render_test)
                    t.set_postfix(test_fitness = avg_fitness)

        return self.stats

    def mutate(self, actor, noise):
        for actor_layer in actor.layers:
            actor_layer.set_weights(list(map(
                lambda w: np.random.normal(loc=w, scale=noise), # parameter noise
                actor_layer.get_weights())))
        return actor

