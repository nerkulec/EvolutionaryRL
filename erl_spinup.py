from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import gym
import time
import core
from tqdm import trange


if torch.cuda.is_available():
  FloatTensor = torch.cuda.FloatTensor
else:
  FloatTensor = torch.FloatTensor

def to_numpy(var):
  return var.cpu().data.numpy()

class RLNN(nn.Module):

  def __init__(self, state_dim, action_dim, max_action):
    super(RLNN, self).__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.max_action = FloatTensor(max_action)

  def set_params(self, params):
    cpt = 0
    for param in self.parameters():
      tmp = np.product(param.size())

      if torch.cuda.is_available():
        param.data.copy_(torch.from_numpy(
          params[cpt:cpt + tmp]).view(param.size()).cuda())
      else:
        param.data.copy_(torch.from_numpy(
          params[cpt:cpt + tmp]).view(param.size()))
      cpt += tmp

  def get_params(self):
    return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                self.parameters()]))

  def get_grads(self):
    return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

  def get_size(self):
    return self.get_params().shape[0]

  def load_model(self, filename, net_name):
    if filename is None:
      return

    self.load_state_dict(
        torch.load('{}/{}.pkl'.format(filename, net_name),
                    map_location=lambda storage, loc: storage))

  def save_model(self, output, net_name):
    torch.save(
        self.state_dict(),
        '{}/{}.pkl'.format(output, net_name)
    )

class Actor(RLNN):
  def __init__(self, state_dim, action_dim, max_action, init=True):
    super(Actor, self).__init__(state_dim, action_dim, max_action)

    self.l1 = nn.Linear(state_dim, 12)
    # self.l2 = nn.Linear(12, 12)
    self.l3 = nn.Linear(12, action_dim)


  def forward(self, x):
    x = torch.tanh(self.l1(x))
    # x = torch.tanh(self.l2(x))
    x = torch.tanh(self.l3(x))*self.max_action

    return x

class Critic(RLNN):
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__(state_dim, action_dim, 1)

    self.l1 = nn.Linear(state_dim + action_dim, 12)
    # self.l2 = nn.Linear(12, 12)
    self.l3 = nn.Linear(12, 1)

  def forward(self, state, action):
    x = torch.tanh(self.l1(torch.cat([state, action], 1)))
    # x = torch.tanh(self.l2(x))
    x = self.l3(x)

    return x

class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
    self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    states = torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32)
    actions = torch.as_tensor(self.act_buf[idxs], dtype=torch.float32)
    rewards = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32)
    next_states = torch.as_tensor(self.obs2_buf[idxs], dtype=torch.float32)
    dones = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32)
    return states, actions, rewards, next_states, dones


class ERL:
  def __init__(self, env, actor, critic, optimizer, doc = None, buffer_size = 10**6, fresh_buffer = False,
          actions_per_epoch = 1, training_steps_per_epoch = 1, gamma = 0.99, polyak = 0.995,
          action_noise = (1, 0.1), num_actors = 10, elite_frac = 0.2, episodes_per_actor = 1,
          episodes_rl_actor = 10, rl_actor_copy_every = 10, mutation_rate = 0.001, mutation_prob = 0.2,
          **kwargs):

    if type(env) is str:
      self.env = gym.make(env)
    else:
      self.env = env
    self.obs_dim = env.observation_space.shape
    self.act_dim = env.action_space.shape[0]
    self.act_limit = env.action_space.high[0]

    self.actor = actor.cuda()
    self.critic = critic.cuda()
    self.target_actor = deepcopy(actor)
    self.target_critic = deepcopy(critic)
    # self.optimizer = optimizer
    self.alg = 'erl'

    for p in self.target_actor.parameters():
      p.requires_grad = False

    for p in self.target_critic.parameters():
      p.requires_grad = False

    self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=buffer_size)

    self.actions_per_epoch = actions_per_epoch
    self.training_steps_per_epoch = training_steps_per_epoch
    self.gamma = gamma
    self.polyak = polyak

    self.mse = nn.MSELoss()
    self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01)
    self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=0.01)

    # self.actors = [deepcopy(self.actor).cuda() for _ in range(num_actors)]
    self.elites = int(elite_frac*num_actors)
    self.episodes_per_actor = episodes_per_actor
    self.episodes_rl_actor = episodes_rl_actor
    self.rl_actor_copy_every = rl_actor_copy_every
    self.mutation_rate = mutation_rate
    self.mutation_prob = mutation_prob

    if callable(action_noise):
      self.action_noise = action_noise
    elif type(action_noise) is tuple: # interpolate linearly
      self.action_noise = lambda x: (1-x)*action_noise[0] + x*action_noise[1]
    else:
      self.action_noise = lambda x: action_noise

    if torch.cuda.is_available():
      self.actor = self.actor.cuda()
      self.target_actor = self.target_actor.cuda()
      self.critic = self.critic.cuda()
      self.target_critic = self.target_critic.cuda()


  def compute_loss_q(self, states, actions, rewards, next_states, done):
    q = self.critic(states.cuda(), actions.cuda())

    with torch.no_grad():
      q_pi_targ = self.target_critic(next_states, self.target_actor(next_states))
      backup = rewards + self.gamma * (1 - done) * q_pi_targ

    loss_q = ((q - backup)**2).mean()

    return loss_q

  def compute_loss_pi(self, states):
    q_pi = self.critic(states.cuda(), self.actor(states.cuda()))
    return -q_pi.mean()

  def update(self, states, actions, rewards, next_states, done):
    self.critic_optimizer.zero_grad()
    loss_q = self.compute_loss_q(states.cuda(),
      actions.cuda(), rewards.cuda(), next_states.cuda(), done.cuda())
    loss_q.backward()
    self.critic_optimizer.step()

    for p in self.critic.parameters():
      p.requires_grad = False

    # Next run one gradient descent step for pi.
    self.actor_optimizer.zero_grad()
    loss_pi = self.compute_loss_pi(states)
    loss_pi.backward()
    self.actor_optimizer.step()

    # Unfreeze Q-network so you can optimize it at next DDPG step.
    for p in self.critic.parameters():
      p.requires_grad = True

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
      for p, p_targ in zip(self.actor.parameters(), self.target_actor.parameters()):
        p_targ.data.mul_(self.polyak)
        p_targ.data.add_((1 - self.polyak) * p.data)
      for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
        p_targ.data.mul_(self.polyak)
        p_targ.data.add_((1 - self.polyak) * p.data)

  def get_action(self, actor, state, noise_scale):
    with torch.no_grad():
      action = actor(torch.as_tensor(state, dtype=torch.float32).cuda()).cpu().numpy()
    action += noise_scale * np.random.randn(self.act_dim)
    return np.clip(action, -self.act_limit, self.act_limit)

  def test_agent(self, episodes):
    for _ in range(episodes):
      state, done, ep_ret, ep_len = self.env.reset(), False, 0, 0
      while not done:
        # Take deterministic actions at test time (noise_scale=0)
        state, reward, done, _ = self.env.step(self.get_action(self.actor, state, 0))
        ep_ret += reward
        ep_len += 1
    print(ep_ret/episodes)

  def train(self, epochs, batch_size = 64, render = False, rl_actor_render = False,
            test_every = 100, render_test = True, render_mode = 'human', **kwargs):

    state = self.env.reset()
    for _ in trange(self.replay_buffer.size, miniters = 1000):
      action = self.env.action_space.sample()
      next_state, reward, done, _ = self.env.step(action)
      self.replay_buffer.store(state, action, reward, next_state, done)
      state = next_state
      if done:
        state = self.env.reset()

    # temporary
    max_ep_len = 200
    update_after = 0
    update_every = 5
    steps_per_epoch = 5

    state, ep_ret, ep_len = self.env.reset(), 0, 0
    for t in trange(epochs):
      action = self.get_action(self.actor, state, self.action_noise(t/epochs))

      next_state, reward, done, _ = self.env.step(action)
      ep_ret += reward
      
      done = False if ep_len == max_ep_len else done
      self.replay_buffer.store(state, action, reward, next_state, done)
      state = next_state

      # End of trajectory handling
      if done or (ep_len == max_ep_len):
        state, ep_ret, ep_len = self.env.reset(), 0, 0

      # Update handling
      if t >= update_after and t % update_every == 0:
        for _ in range(update_every):
          states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(batch_size)
          self.update(states, actions, rewards, next_states, dones)

      # End of epoch handling
      if (t+1) % steps_per_epoch == 0:
        # epoch = (t+1) // steps_per_epoch
        self.test_agent(1)
