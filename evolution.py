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
        for _ in range(int(mut_frac*product(param.shape))):
            ixs = tuple(randint(0, x-1) for x in param.shape)
            if random() < super_mut:
                param.data[ixs] = param.data[ixs]*randn()*noise*100
            elif random() < reset_mut:
                param.data[ixs] = randn()
            else:
                param.data[ixs] = param.data[ixs]*randn()*noise
    return actor

def evolution(actors, rewards, mutation_rate, num_elites=1):
    rank = np.argsort(rewards)[::-1] # best on the front
    elites = [deepcopy(actors[i]) for i in rank[:num_elites]]
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