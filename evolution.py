import torch
import numpy as np
from copy import deepcopy
from functools import reduce
import operator
from random import randint, random
from numpy.random import randn
import fastrand

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


def crossover_inplace(gene1, gene2):
    for param1, param2 in zip(gene1.parameters(), gene2.parameters()):

        # References to the variable tensors
        W1 = param1.data
        W2 = param2.data

        if len(W1.shape) == 2: #Weights no bias
            num_variables = W1.shape[0]
            # Crossover opertation [Indexed by row]
            num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
            for i in range(num_cross_overs):
                receiver_choice = random.random()  # Choose which gene to receive the perturbation
                if receiver_choice < 0.5:
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                    W1[ind_cr, :] = W2[ind_cr, :]
                else:
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                    W2[ind_cr, :] = W1[ind_cr, :]

        elif len(W1.shape) == 1: #Bias
            num_variables = W1.shape[0]
            # Crossover opertation [Indexed by row]
            num_cross_overs = fastrand.pcg32bounded(num_variables)  # Lower bounded on full swaps
            for i in range(num_cross_overs):
                receiver_choice = random.random()  # Choose which gene to receive the perturbation
                if receiver_choice < 0.5:
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                    W1[ind_cr] = W2[ind_cr]
                else:
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                    W2[ind_cr] = W1[ind_cr]

# def mutate(actor, noise=0.1):
#     device = 'cpu'
#     for param in actor.parameters():
#         param.data.add_(torch.randn_like(param.data, device=device)*noise)
#     return actor

last_best_reward = -1e10
best_param = None

def evolution(actors, rewards, mutation_rate, num_elites=1):
    global last_best_reward, best_param
    rank = np.argsort(rewards)[::-1] # best on the front
    elites = [deepcopy(actors[i]) for i in rank[:num_elites]]
    best_reward = rewards[rank[0]]
    param = list(elites[0].parameters())[0][0].detach().numpy()
    worsened = False
    if best_reward < last_best_reward:
        if (best_param - param).sum() != 0:
            print(f'params mutated: {best_reward} < {last_best_reward}')
            last_best_reward = best_reward
            worsened = True
        else:
            print(f'params not mutated: {best_reward} < {last_best_reward}')
    if best_reward > last_best_reward:
        last_best_reward = best_reward
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
    return elites+rest, dict(worsened = worsened)