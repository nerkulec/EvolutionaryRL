import torch
import numpy as np
from copy import deepcopy

def mutate(actor, noise=0):
    device = 'cpu'
    for param in actor.parameters():
        param.data.add_(torch.randn_like(param.data, device=device)*noise)
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