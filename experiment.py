# %%
env_name = 'CarEnv-v0'
alg_name = 'ERL3'
exp_name = 'torch_test-more-explore'

from documenter import Documenter
doc = Documenter(env_name, alg_name, exp_name,
    # algorithm
    episodes_per_actor = 1,
    episodes_rl_actor = 5,
    training_steps_per_epoch = 10,
    num_actors = 40, elite_frac = 0.05,
    mutation_prob  =  0.8,
    mutation_rate  =  0.5,
    action_noise  =  0,
    buffer_size  =  10**5,
    fresh_buffer  =  False,
    polyak  =  0.99,

    # optimizer
    lr = 0.02,

    # training
    epochs = 4000,
    batch_size = 2048,
    test_every = 5,
    render_test = False
)
doc.add_folder('maps')
doc.add_file('erl.py')
doc.add_file('es.py')
doc.add_file('experiment.py')
doc.add_file('buffer.py')
doc.add_file('car_env.py')
doc.add_file('car.py')
doc.add_file('documenter.py')
doc.add_file('racing_track.py')
doc.add_file('util.py')

doc.save_files()
doc.save_settings()

import numpy as np
np.random.seed(hash(doc)%(2**32))
# %%
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam, SGD

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange
from datetime import datetime
import pathlib

# from erl import ERL
from erl_torch import ERL, Actor, Critic

import sys
sys.path.append('~/RLCar/car_env')
import gym
import car_env
env = gym.make(env_name)

# %%
state_size = int(np.prod(env.observation_space.shape))
action_size = int(np.prod(env.action_space.shape))
# if alg_name != 'SimpleES':
#     try:
#         actor  = tf.keras.models.load_model(doc.get_model_path('actor'))
#         critic = tf.keras.models.load_model(doc.get_model_path('critic'))
#         print('Models loaded from disk')

#     except OSError:
#         try:
#             action_high = float(env.action_space.high)
#         except:
#             action_high = float(env.action_space.high[0])

#         actor = Sequential([
#             Input(state_size),
#             Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
#             Dense(action_size, activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
#             Lambda(lambda x: x*action_high)
#         ])
#         critic = Sequential([
#             Input(state_size+action_size),
#             Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
#             Dense(1)
#         ])
#         print('Fresh models created')
actor = Actor(state_size, action_size, env.action_space.high)
critic = Critic(state_size, action_size)

# %%
# adam = Adam()
sgd = SGD(learning_rate=doc.settings['lr'])
alg = ERL(env, actor, critic, sgd, doc = doc, **doc.settings)
# %%
render_mode = 'trajectories'
env.render(mode=render_mode)
try:
    stats = alg.train(render_mode=render_mode, **doc.settings)

except (KeyboardInterrupt, SystemExit) as e:
    print('Interrupted')
    stats = alg.stats

env.close()

# doc.save_buffer(alg.buffer)
# doc.save_models(alg)

print(f'Buffer overwritten {alg.buffer.i/alg.buffer.size:.1f} times')
# %%
plt.plot(stats['critic_loss'])
plt.suptitle('Critic loss')
plt.xlabel('Train steps')
doc.save_fig(plt, 'critic_loss')
# %%
plt.plot(stats['actor_loss'])
plt.suptitle('Actor loss')
plt.xlabel('Train steps')
doc.save_fig(plt, 'actor_loss')
# %%
critic_pred = np.array(stats['critic_pred'])
mean = np.mean(critic_pred, axis=1)
std_dev = np.sqrt(np.mean(np.square(critic_pred-mean.reshape(-1, 1)), axis=1))
plt.plot(mean)
plt.fill_between(np.arange(critic_pred.shape[0]), mean-std_dev*1.5, mean+std_dev*1.5, alpha = 0.5)
plt.suptitle('Critic prediction')
plt.xlabel('Train steps')
doc.save_fig(plt, 'critic_pred')
# %%
actor_pred = np.array(stats['actor_pred'])
actor_pred_x = actor_pred[:,:,0]
actor_pred_y = actor_pred[:,:,1]
mean = np.mean(actor_pred, axis=1)
mean_x = mean[:,0]
mean_y = mean[:,1]
std_dev_x = np.sqrt(np.mean(np.square(actor_pred_x-mean_x.reshape(-1, 1)), axis=1))
std_dev_y = np.sqrt(np.mean(np.square(actor_pred_y-mean_y.reshape(-1, 1)), axis=1))
plt.plot(mean_x)
plt.plot(mean_y)
plt.fill_between(np.arange(actor_pred_x.shape[0]), mean_x-std_dev_x*1.5, mean_x+std_dev_x*1.5, alpha = 0.5)
plt.fill_between(np.arange(actor_pred_y.shape[0]), mean_y-std_dev_y*1.5, mean_y+std_dev_y*1.5, alpha = 0.5)
plt.suptitle('Actor prediction')
plt.xlabel('Train steps')
doc.save_fig(plt, 'actor_pred')
# %%
critic_weights = stats['critic_weights']
fig, ax = plt.subplots()
for i in range(len(critic_weights[0])):
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot([w[i].flatten() for w in critic_weights], color = color)
plt.suptitle('Critic weights')
plt.xlabel('Train steps')
doc.save_fig(plt, 'critic_weights')

# %%
actor_weights = stats['actor_weights']
fig, ax = plt.subplots()
for i in range(len(actor_weights[0])):
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot([w[i].flatten() for w in actor_weights], color = color)
plt.suptitle('Actor weights')
plt.xlabel('Train steps')
doc.save_fig(plt, 'actor_weights')

# %%
actors_rewards = np.array(stats['actors_rewards'])
mean = np.mean(actors_rewards, axis=1)
low  = np.min(actors_rewards, axis=1)
high = np.max(actors_rewards, axis=1)

plt.plot(actors_rewards)

plt.suptitle('Evolutionary actors rewards')
plt.xlabel('Epochs')

doc.save_fig(plt, 'actors_reward')
# %%
