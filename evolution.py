import torch
import numpy as np
from copy import deepcopy
from functools import reduce
import operator
from random import randint, random
from numpy.random import randn

def product(xs):
    return reduce(operator.mul, xs, 1)

def mutate(actor, noise=0.1, mut_frac=0.1, super_mut=0.05, reset_mut=0.05):
    device = 'cpu'
    for param in actor.parameters():
        if len(param.shape) == 2:
            for _ in range(int(mut_frac*product(param.shape))):
                ixs = tuple(randint(0, x-1) for x in param.shape)
                if random() < super_mut:
                    param.data[ixs] = param.data[ixs]*randn()*noise*100
                elif random() < reset_mut:
                    param.data[ixs] = randn()
                else:
                    param.data[ixs] = param.data[ixs]*randn()*noise
    return actor

# def mutate(actor, noise=0.1):
#     device = 'cpu'
#     for param in actor.parameters():
#         param.data.add_(torch.randn_like(param.data, device=device)*noise)
#     return actor

all_time_best_reward = -1e10
best_param = None

def evolution(actors, rewards, mutation_rate, num_elites=1):
    global all_time_best_reward, best_param
    rank = np.argsort(rewards)[::-1] # best on the front
    elites = [deepcopy(actors[i]) for i in rank[:num_elites]]
    best_reward = rewards[rank[0]]
    param = list(elites[0].parameters())[0][0].detach().numpy()
    if best_reward < all_time_best_reward:
        if (best_param - param).sum() != 0:
            print(f'params mutated: {best_reward} < {all_time_best_reward}')
        else:
            print(f'params not mutated: {best_reward} < {all_time_best_reward}')
    if best_reward > all_time_best_reward:
        all_time_best_reward = best_reward
        best_param = param.copy()

    rest = []
    for _ in range(len(actors)-num_elites): # tournament selection
        a = np.random.choice(rank) # select from all
        b = np.random.choice(rank)
        if rewards[a] > rewards[b]: # maybe do crossover here
            winner = actors[a]
        else:
            winner = actors[b]
        winner = deepcopy(winner)
        mutate(winner, mutation_rate)
        rest.append(winner)
    return elites+rest