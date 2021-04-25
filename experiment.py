# %%

from documenter import Documenter
import numpy as np
import math
import os
import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange
from datetime import datetime
import pathlib

from erl_spinup3 import ERL
from td3_spinup3 import TD3
from sac_spinup import SAC

alg_by_name = dict(
    ERL = ERL,
    TD3 = TD3,
    SAC = SAC
)

import gym
# %%
envs = [
    'rl_car:RLCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0'
]
algs = ['ERL', 'TD3', 'SAC']
num_actors = [0, 4, 10, 40]
mutation_rates = [0.01, 0.04, 0.1]
buffer_size = [10**6, 10**5]
alg_specific = {
    'TD3': dict(
        noise_clip = [0.1, 0.2, 0.5]
    )
}
variant_specific = {

}

# %%
doc = Documenter(
    env_name = 'rl_car:RLCar-v0',
    alg_name = 'TD3',
    exp_name = 'TD3 test',
    seed = 0,

    episodes_rl_actor = 5,
    rl_actor_copy_every = 5,
    steps_per_epoch = 4000,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    num_actors = 0,
    elites = 1,
    mutation_prob = 1,
    mutation_rate = 0.02,
    act_noise = 0.01,
    buffer_size = 10**5,
    gamma = 0.99,
    polyak = 0.995,
    noise_clip = 0.5,
    policy_delay = 2,
    target_noise = 0.2,

    # model
    hidden_sizes = [24],

    # optimizer
    lr = 0.001,

    # training
    epochs = 10,
    batch_size = 256,
    test_every = 5,
    render_mode = 'trajectories',
    render_test = False
)
doc.save_settings()

alg = TD3(env=doc.env_name, **doc.settings)
try:
    stats = alg.train(**doc.settings)

except (KeyboardInterrupt, SystemExit) as e:
    print('Interrupted')
    stats = alg.stats
    alg.env.close()

doc.draw_plots(stats)
# %%
