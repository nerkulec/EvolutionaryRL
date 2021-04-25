from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from tqdm import tqdm, trange
import sys
sys.path.append('/home/bartek/spinningup')
import spinup.algos.pytorch.ddpg.core as core
from spinup.utils.logx import EpochLogger

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
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class DDPG:
    def __init__(self, env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, replay_size=int(1e4), gamma=0.99, episodes_rl_actor = 10,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, act_noise=0.1, start_steps=int(10e4), update_every=50,
        num_test_episodes=10, logger_kwargs=dict(), update_after=1000):

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        if type(env) is str:
            env = gym.make(env)
        self.env = env

        self.steps_per_epoch = steps_per_epoch
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.episodes_rl_actor = episodes_rl_actor
        self.start_steps = start_steps
        self.update_every = update_every
        self.update_after = update_after

        # env.render('trajectories')
        self.env.opacity = 64
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        # replay_buffer = Buffer(replay_size, env)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
        # logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()


    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Record things
        self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        self.env.color = (0, 0, 255)
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d or (ep_len == self.env._max_episode_steps)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
                self.env.render('trajectories')
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train(self, epochs, batch_size):
        # Prepare for interaction with environment
        self.env.color = (255, 255, 255)
        self.env.opacity = 30
        self.env.render('trajectories')
        start_time = time.time()
        o, ep_len = self.env.reset(), 0

        for i in trange(self.start_steps, miniters = 1000):
            a = self.env.action_space.sample()
            o2, r, d, _ = self.env.step(a)
            d = False if ep_len==self.env._max_episode_steps else d
            self.replay_buffer.store(o, a, r, o2, d)
            o = o2
            if d or (ep_len == self.env._max_episode_steps):
                o, ep_len = self.env.reset(), 0

        # Main loop: collect experience in env and update/log each epoch
        self.env.opacity = 64
        t = 0
        for epoch in range(epochs):
            self.env.color = (255, 0, 0)
            o, d, ep_len, ep_ret = self.env.reset(), False, 0, 0
            for _ in range(self.steps_per_epoch):
                t += 1
                a = self.get_action(o, self.act_noise)
                o2, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1
                d = False if ep_len==self.env._max_episode_steps else d
                self.replay_buffer.store(o, a, r, o2, d)
                o = o2
                if d or (ep_len == self.env._max_episode_steps):
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, d, ep_len, ep_ret = self.env.reset(), False, 0, 0
                    self.env.render('trajectories')

                # Update
                if t >= self.update_after and t % self.update_every == 0:
                    for _ in range(self.update_every):
                        data = self.replay_buffer.sample_batch(batch_size)
                        self.update(data)
            self.test_agent()

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('TestEpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('TestEpLen', average_only=True)
            self.logger.log_tabular('QVals', with_min_and_max=True)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossQ', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()
            self.env.render('trajectories')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg = DDPG(gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed,
         logger_kwargs=logger_kwargs)
    ddpg.train(epochs = args.epochs, batch_size = 256)
