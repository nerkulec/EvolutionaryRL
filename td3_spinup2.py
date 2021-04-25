from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
import gym
import time
from tqdm import tqdm, trange
import spinup.algos.pytorch.ddpg.core as core
from spinup.utils.logx import EpochLogger
import random
from collections import defaultdict
import itertools

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

def to_numpy(var):
    return var.cpu().data.numpy()

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = FloatTensor(np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32))
        self.obs2_buf = FloatTensor(np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32))
        self.act_buf = FloatTensor(np.zeros(core.combined_shape(size, act_dim), dtype=np.float32))
        self.rew_buf = FloatTensor(np.zeros(size, dtype=np.float32))
        self.done_buf = FloatTensor(np.zeros(size, dtype=np.float32))
        self.ptr, self.size, self.max_size = 0, 0, size
        self.i = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = FloatTensor(obs)
        self.obs2_buf[self.ptr] = FloatTensor(next_obs)
        self.act_buf[self.ptr] = FloatTensor(act)
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        self.i += 1

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: v for k,v in batch.items()}

class TD3:
    def __init__(self, env, hidden_sizes=(256,256), activation=nn.ReLU, seed=0, 
        rl_actor_steps=1200, evo_actor_steps=400, replay_size=int(1e5), gamma=0.99,
        steps_per_epoch=4000, render_mode='human',
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, act_noise=0.01, start_steps=int(1e4), update_every=50,
        num_test_episodes=1, logger_kwargs=dict(), update_after=1000, rl_actor_copy_every=5,
        num_actors=10, elite_frac=0.2, mutation_rate=0.1, mutation_prob=0.8,
        noise_clip=0.5, policy_delay=2, target_noise=0.2,
        doc=None, **kwargs):

        self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.stats = defaultdict(list) # switch to tensorboard

        if type(env) is str:
            env = gym.make(env)
        self.env = env

        self.steps_per_epoch = steps_per_epoch
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.start_steps = start_steps
        self.update_every = update_every
        self.update_after = update_after

        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.target_noise = target_noise

        self.env.opacity = 64
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]

        # Create actor and critic module and target networks
        self.pi = core.MLPActor(self.obs_dim, self.act_dim, hidden_sizes, activation, self.act_limit).cuda()
        self.q1 = core.MLPQFunction(self.obs_dim, self.act_dim, hidden_sizes, activation).cuda()
        self.q2 = core.MLPQFunction(self.obs_dim, self.act_dim, hidden_sizes, activation).cuda()

        self.pi_targ = deepcopy(self.pi)
        self.q1_targ = deepcopy(self.q1)
        self.q2_targ = deepcopy(self.q2)

        self.params = itertools.chain(self.pi.parameters(), self.q1.parameters(), self.q2.parameters())
        self.targ_params = itertools.chain(self.pi_targ.parameters(), self.q1_targ.parameters(), self.q2_targ.parameters())

        self.num_actors = num_actors
        self.actors = [self.mutate(deepcopy(self.pi), 0.1) for _ in range(num_actors)]
        self.elites = int(elite_frac*num_actors)
        self.rl_actor_copy_every = rl_actor_copy_every
        self.mutation_rate = mutation_rate
        self.mutation_prob = mutation_prob

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.targ_params:
            p.requires_grad = False
        for actor in self.actors:
            for p in actor.parameters():
                p.requires_grad = False
        
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

    # Set up function for computing TD3 Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.pi_targ(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            # Target Q-values
            q1_pi_targ = self.q1_targ(o2, a2)
            q2_pi_targ = self.q2_targ(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy())

        self.stats['critic_loss'].append(to_numpy(loss_q))
        # self.stats['critic_pred'].append(to_numpy(q.flatten()))
        # self.stats['critic_weights'].append(list(map(to_numpy, self.ac.q.parameters())))

        return loss_q, loss_info

    def compute_loss_pi(self, data):
        o = data['obs']
        a = self.pi(o)
        q_pi = self.q1(o, a)
        minus_q_pi_mean = -q_pi.mean()

        self.stats['actor_loss'].append(to_numpy(minus_q_pi_mean))
        self.stats['actor_pred'].append(to_numpy(a))
        self.stats['actor_weights'].append(list(map(to_numpy, self.pi.parameters())))
        return minus_q_pi_mean

    def update(self, data, timer):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % self.policy_delay == 0:
            # Freeze Q-network so you don't waste computational effort 
            # computing gradients for it during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next TD3 step.
            for p in self.q_params:
                p.requires_grad = True

            # Record things
            self.logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.params, self.targ_params):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, actor, obs, noise_scale=0):
        obs = FloatTensor(obs)
        with torch.no_grad():
            action = actor(obs)
        action = action.cpu().numpy()
        if noise_scale != 0:
            action += noise_scale * np.random.randn(self.act_dim)
        return np.clip(action, -self.act_limit, self.act_limit)

    def mutate(self, actor, noise):
        actor = actor.cpu()
        for param in actor.parameters():
            param.data.copy_(FloatTensor(np.random.normal(loc=param.data, scale=noise)))
        actor = actor.cuda()
        return actor

    def test_agent(self, render=False, render_mode='human'):
        self.env.color = (0, 255, 0)
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d or (ep_len == self.env._max_episode_steps)):
                o, r, d, _ = self.env.step(self.get_action(self.pi, o, 0))
                ep_ret += r
                ep_len += 1
                if render:
                    self.env.render(render_mode)
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def evolution(self, rewards):
        rank = np.argsort(rewards)[::-1] # best on the front
        elites = [deepcopy(self.actors[i]) for i in rank[:self.elites]]
        rest = []
        for _ in range(len(self.actors)-self.elites): # tournament selection
            a = np.random.choice(rank) # select from all
            b = np.random.choice(rank)
            if rewards[a] > rewards[b]: # maybe do crossover here
                winner = self.actors[a]
            else:
                winner = self.actors[b]
            # winner = deepcopy(winner)
            if random.random() < self.mutation_prob:
                winner = self.mutate(winner, self.mutation_rate)
            rest.append(winner)
        self.actors = elites+rest

    def train(self, epochs, batch_size, render_mode='human', render_test=False, **kwargs):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        self.env.render(render_mode)
        self.env.color = (255, 255, 255)
        # Start steps
        for t in trange(self.start_steps, miniters = 1000):
            a = self.env.action_space.sample()
            o2, r, d, _ = self.env.step(a)
            d = False if ep_len==self.env._max_episode_steps else d
            self.replay_buffer.store(o, a, r, o2, d)
            o = o2
            if d or (ep_len == self.env._max_episode_steps):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

        actor_num = -1
        evo_rewards = []
        # Main loop: collect experience in env and update/log each epoch
        with tqdm(total=epochs) as pbar:
            for t in range(self.start_steps, total_steps):
                if actor_num == -1:
                    self.env.color = (255, 0, 0)
                    actor = self.pi
                else:
                    self.env.color = (0, 0, 255)
                    actor = self.actors[actor_num]
                a = self.get_action(actor, o, self.act_noise)

                # Step the self.env
                o2, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1
                d = False if ep_len==self.env._max_episode_steps else d

                # Store experience to replay buffer
                self.replay_buffer.store(o, a, r, o2, d)
                o = o2

                # End of trajectory handling
                if d or (ep_len == self.env._max_episode_steps):
                    if actor_num == -1:
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                        self.stats['reward'].append(ep_ret)
                    else:
                        self.logger.store(EvoEpRet=ep_ret, EvoEpLen=ep_len)
                        evo_rewards.append(ep_ret)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    if self.num_actors > 0:
                        actor_num += 1
                    if(actor_num >= len(self.actors)):
                        self.evolution(rewards=evo_rewards)
                        self.stats['actors_rewards'].append(list(sorted(evo_rewards)))
                        actor_num = -1
                        evo_rewards = []
                    self.env.render(render_mode)

                # Update handling
                if t >= self.update_after and t % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(batch_size)
                        self.update(data=batch, timer=j)

                # End of epoch handling
                if (t+1) % self.steps_per_epoch == 0:
                    pbar.update()
                    epoch = (t+1) // self.steps_per_epoch

                    # Test the performance of the deterministic version of the agent.
                    self.test_agent(render_test, render_mode)

                    # Log info about epoch
                    self.logger.log_tabular('Epoch', epoch)
                    self.logger.log_tabular('EpRet', with_min_and_max=True)
                    if self.num_actors > 0:
                        self.logger.log_tabular('EvoEpRet', with_min_and_max=True)
                    self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                    self.logger.log_tabular('EpLen', average_only=True)
                    if self.num_actors > 0:
                        self.logger.log_tabular('EvoEpLen', with_min_and_max=True)
                    self.logger.log_tabular('TestEpLen', average_only=True)
                    self.logger.log_tabular('TotalEnvInteracts', t)
                    # self.logger.log_tabular('QVals', with_min_and_max=True)
                    self.logger.log_tabular('LossPi', average_only=True)
                    self.logger.log_tabular('LossQ', average_only=True)
                    self.logger.log_tabular('Time', time.time()-start_time)
                    self.logger.dump_tabular()
                    self.env.render(render_mode)
        return self.stats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--render_mode', type=str, default='human')
    parser.add_argument('--evo_actors', type=int, default=10)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    td3 = TD3(gym.make(args.env),
         hidden_sizes=[args.hid]*args.l, 
         gamma=args.gamma, seed=args.seed, render_mode=args.render_mode,
         num_actors=args.evo_actors,
         logger_kwargs=logger_kwargs)
    td3.train(epochs = args.epochs, batch_size = 256)
