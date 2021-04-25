import numpy as np
import torch
import spinup.algos.pytorch.ddpg.core as core

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

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
