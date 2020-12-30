import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import clone_model
from tqdm.auto import tqdm, trange
import pickle

from buffer import Buffer

## IGNORE THIS FILE

class ERL:
    def __init__(self, Alg, env, actor, critic, optimizer, pop_size = 10,
        elite_frac = 0.1, eval_episodes = 10, mut_prob = 0.2, mut_scale = 0.1,
        sync_period = 10, **kwargs):
        self.alg = Alg(env, actor, critic, optimizer, **kwargs)
        self.actors = [tf.keras.models.clone_model(actor) for _ in range(pop_size)]
        # self.envs = [self.alg.env.sim.get_state() for _ in range(pop_size)]
        self.elites = int(pop_size*elite_frac)
        self.eval_episodes = eval_episodes
        self.mut_prob = mut_prob
        self.mut_scale = mut_scale
        self.sync_period = sync_period

    def mutate(self, actor): # add layer normalization
        for actor_layer in actor.layers:
            actor_layer.set_weights(list(map(
                lambda alw: np.random.normal(alw, self.mut_scale),
                actor_layer.get_weights())))

    def evolve(self, noise):
        selected = []
        fitness = [self.eval(actor, noise, self.eval_episodes)
                   for actor in self.actors]
        idx = np.argsort(fitness)
        for i in range(self.elites):
            selected.append(idx[i])
        for i in range(len(self.actors)-self.elites):
            a = np.random.choice(len(self.actors))
            b = np.random.choice(len(self.actors))
            # Add crossover
            selected.append(a if fitness[a] > fitness[b] else b)
        self.actors = [clone_model(self.actors[i]) for i in selected]
        for actor in self.actors:
            if np.random.rand() < self.mut_prob:
                self.mutate(actor)

    def train(self, epochs, batch_size = 64):        
        if not self.alg.buffer.full:
            state = self.alg.env.reset()
            print("Filling up buffer")
            for _ in tqdm(range(self.alg.buffer.size)):
                action = self.alg.env.action_space.sample()
                state = self.alg.simulate(state, action)
            print("Buffer filled")
            self.alg.buffer.save_to_file()
        
        print("Now training")
        with trange(epochs) as t:
            for epoch in t:
                self.evolve(epoch/epochs)
                avg_fitness = self.eval(self.alg.actor, self.alg.noise(epoch/epochs), 10, render = True)
                t.set_postfix(avg_fitness = avg_fitness)
                self.alg.train_step(batch_size)

                if epoch % self.sync_period:
                    i = np.argmin([self.eval(actor, self.alg.noise(epoch/epochs),
                        self.eval_episodes) for actor in self.actors])
                    self.actors[i] = clone_model(self.alg.actor)

    def eval(self, actor, noise, num_episodes, render=False):
        env = self.alg.env
        fitness = 0
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = actor(np.array([state])).numpy()[0]
                action = np.clip(np.random.normal(loc=action, scale=noise),
                    env.action_space.low, env.action_space.high)
                next_state, reward, done, _ = env.step(action)
                self.alg.buffer.store(state, action, reward, next_state, done)
                if render:
                    self.alg.env.render()
                state = next_state
                fitness += reward
        return fitness/num_episodes

        
    def test(self, num_episodes):
        self.alg.test(num_episodes)


