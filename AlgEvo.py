# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import math
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange
from datetime import datetime

from ddpg import DDPG
from erl import ERL


# %%
# env = 'InvertedPendulum-v2'
env = 'CartPole-v1'
# alg = 'ERL'
alg = 'DDPG'
name = 'test'

try:
    actor  = tf.keras.models.load_model(f'models/{env}-{alg}-{name}-actor')
    critic = tf.keras.models.load_model(f'models/{env}-{alg}-{name}-critic')
    print('Models loaded from disk')

except OSError:
    actor = Sequential([
        Input(4),
        Dense(12, activation='relu'),
        Dense(1, activation='tanh'),
        Lambda(lambda x: 3*x)
    ])
    critic = Sequential([
        Input(5),
        Dense(12, activation='relu'),
        Dense(1)
    ])
    print('Fresh models created')


# %%
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
test_log_dir = 'logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# %%
adam = Adam()
# x = epoch/epochs \in [0, 1]
action_noise = lambda x: 10**(-2*x-2) # from 0.01 to 0.0001 logarythmically
ddpg = DDPG(env, actor, critic, adam, name = name, actions_per_epoch=4, training_steps_per_epoch=4,
    action_noise = action_noise, train_summary_writer = train_summary_writer, test_summary_writer = test_summary_writer)


# %%
try:
    stats = ddpg.train(epochs = 10000, batch_size=256, test_every=200)

except (KeyboardInterrupt, SystemExit) as e:
    print('Interrupted')
    stats = ddpg.stats
    
ddpg.buffer.save_to_file()
actor.save( f'models/{env}-{alg}-{name}-actor')
critic.save(f'models/{env}-{alg}-{name}-critic')
ddpg.env.close()


# %%
plt.plot(stats['reward'])
plt.suptitle('Train reward')
plt.xlabel('full episodes')
plt.savefig(f'imgs/train_reward_lognoise_{name}.jpg')


# %%
plt.plot(stats['test_reward'])
plt.suptitle('Test reward')
plt.xlabel('Hundreds of epochs')
plt.savefig(f'imgs/test_reward_lognoise_{name}.jpg')


# %%



