# %%
env_name = 'RLCar-v0'
alg_name = 'ERL2'
name = 'base'

import numpy as np
np.random.seed(hash(f'{env_name}-{alg_name}-{name}')%(2**32))


# %%
import math
import gym
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

from erl import ERL
from car_env import CarEnv
# %%
pathlib.Path('buffers').mkdir(parents=True, exist_ok=True)
pathlib.Path('models' ).mkdir(parents=True, exist_ok=True)

env = None
if env_name == 'RLCar-v0':
    env = CarEnv(file_name = 'maps/map4.txt', num_rays = 12, draw_rays=False)
else:
    env = gym.make(env_name)


# %%
state_size = int(np.prod(env.observation_space.shape))
action_size = int(np.prod(env.action_space.shape))
if alg_name != 'SimpleES':
    try:
        actor  = tf.keras.models.load_model(f'models/{env_name}/{alg_name}-{name}-actor')
        critic = tf.keras.models.load_model(f'models/{env_name}/{alg_name}-{name}-critic')
        print('Models loaded from disk')

    except OSError:
        try:
            action_high = float(env.action_space.high)
        except:
            action_high = float(env.action_space.high[0])

        # one-layer perceptrons:
        actor = Sequential([
            Input(state_size),
            # Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            Dense(action_size, activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            Lambda(lambda x: x*action_high)
        ])
        critic = Sequential([
            Input(state_size+action_size),
            # Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            Dense(1)
        ])
        print('Fresh models created')

# %%
# adam = Adam()
sgd = SGD(learning_rate=0.01)
alg = ERL(env, actor, critic, sgd, name = name, episodes_per_actor=1, episodes_rl_actor=5,
    training_steps_per_epoch=10, num_actors=40, elite_frac=0.05, mutation_prob = 0.8,
    action_noise = 0, mutation_rate = 0.2, buffer_size = 10**5, polyak = 0.9)
# %%
env.render(mode='trajectories')
try:
    stats = alg.train(epochs=4000, batch_size=256, test_every=5, render_test=False, render_mode='trajectories')

except (KeyboardInterrupt, SystemExit) as e:
    print('Interrupted')
    stats = alg.stats

# alg.buffer.save_to_file()
# alg.save_models()
# %%
plt.plot(stats['critic_loss'])
plt.suptitle('Critic loss')
plt.xlabel('Train steps')
directory = f'imgs/{alg_name}/{env_name}'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
plt.savefig(f'{directory}/critic_loss_{name}.jpg')

# %%
plt.plot(stats['actor_loss'])
plt.suptitle('Actor loss')
plt.xlabel('Train steps')
directory = f'imgs/{alg_name}/{env_name}'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
plt.savefig(f'{directory}/actor_loss_{name}.jpg')
# %%
critic_pred = np.array(stats['critic_pred'])
mean = np.mean(critic_pred, axis=1)
std_dev = np.sqrt(np.mean(np.square(critic_pred-mean.reshape(-1, 1)), axis=1))
plt.plot(mean)
plt.fill_between(np.arange(critic_pred.shape[0]), mean-std_dev*1.5, mean+std_dev*1.5, alpha = 0.5)
plt.suptitle('Critic prediction')
plt.xlabel('Train steps')
directory = f'imgs/{alg_name}/{env_name}'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
plt.savefig(f'{directory}/critic_pred_{name}.jpg')
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
directory = f'imgs/{alg_name}/{env_name}'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
plt.savefig(f'{directory}/actor_pred_{name}.jpg')
# %%
critic_weights = np.array(stats['critic_weights'])
s = critic_weights.shape
critic_weights = critic_weights.reshape((s[0], s[1]))
plt.plot(critic_weights)
plt.suptitle('Critic weights')
plt.xlabel('Train steps')
directory = f'imgs/{alg_name}/{env_name}'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
plt.savefig(f'{directory}/critic_weights_{name}.jpg')

# %%
actor_weights = np.array(stats['actor_weights'])
s = actor_weights.shape
actor_weights = actor_weights.reshape((s[0], s[1]*2))
plt.plot(actor_weights)
plt.suptitle('Actor weights')
plt.xlabel('Train steps')
directory = f'imgs/{alg_name}/{env_name}'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
plt.savefig(f'{directory}/actor_weights_{name}.jpg')

# %%
actors_rewards = np.array(stats['actors_rewards'])
mean = np.mean(actors_rewards, axis=1)
low  = np.min(actors_rewards, axis=1)
high = np.max(actors_rewards, axis=1)

plt.plot(actors_rewards)
directory = f'imgs/{alg_name}/{env_name}'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

plt.suptitle('Evolutionary actors rewards')
plt.xlabel('Epochs')

plt.savefig(f'{directory}/actors_reward_{name}.jpg')
# %%
