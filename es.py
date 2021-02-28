import numpy as np
import math
from tqdm.auto import tqdm, trange
from collections import defaultdict

class SimpleES:
    def __init__(self, env, num_actors, num_features, num_outputs, activation_function=lambda x: x, mutation_rate=0.001, elite_frac=0.2):
        self.env = env
        self.num_actors = num_actors
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.activation_function = activation_function
        self.mutation_rate = mutation_rate
        self.actors = np.random.normal(size=(num_actors, 1+num_features, num_outputs))
        self.elites = math.floor(elite_frac*num_actors)

    def get_action(self, actor_num, state):
        actor = self.actors[actor_num]
        obs = np.concatenate([[1], state], axis=0)
        val = np.sum(actor.T*obs, axis=1)
        return self.activation_function(val)

    def mutate(self, without_elites = True):
        epsilon = np.random.normal(0, self.mutation_rate, size=self.actors.shape)
        if without_elites:
            epsilon[:self.elites,:] = 0
        self.actors = self.actors + epsilon

    def train(self, epochs, render = False, test_every = 100, render_test = True, batch_size=None):
        self.stats = defaultdict(list)

        print("Now training")
        with trange(epochs) as t:
            for epoch in t:
                rewards = []
                for actor_num in range(self.num_actors):
                    done = False
                    state = self.env.reset()

                    actor_reward = 0
                    while not done:
                        action = self.get_action(actor_num, state)
                        next_state, reward, done, _ = self.env.step(action)
                        state = next_state
                        actor_reward += reward
                        if render:
                            self.env.render()
                    rewards.append(actor_reward)
                self.stats['actors_rewards'].append(list(sorted(rewards)))
                rank = np.argsort(rewards)
                elites = [self.actors[i] for i in rank[-self.elites:]]
                rest = []
                for _ in range(self.num_actors-self.elites): # tournament selection
                    a = np.random.choice(rank)
                    b = np.random.choice(rank)
                    if rewards[a] > rewards[b]: # maybe do crossover here
                        rest.append(self.actors[a])
                    else:
                        rest.append(self.actors[b])
                self.actors = np.array(elites+rest) # elites on the front

                if (epoch+1) % test_every == 0:
                    self.test(render = render_test, render_only_elites = True)

                self.mutate()
        return self.stats

    def test(self, render = False, render_only_elites = True): # only difference is rendering and no evolution
        fitness = 0
        num = render_only_elites and self.elites or self.num_actors
        for actor_num in range(num):
            state = self.env.reset()
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(actor_num, state)
                state, reward, done, _ = self.env.step(action)
                fitness += reward
        return fitness/num
