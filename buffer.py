import numpy as np
from tqdm.auto import tqdm, trange
import pickle

def get_buffer(size, env, name='1'):
    try:
        file_name = f'buffers/{env.spec.id}-{name}.pkl'
        with open(file_name, 'rb') as f:
            print('Buffer loaded from disk')
            return pickle.load(f)

    except FileNotFoundError:
        print('Fresh buffer created')
        return Buffer(size, env, name)

class Buffer:
    def __init__(self, size, env, name='1'):
        self.size = size
        self.env = env
        self.env_id = env.spec.id
        self.name = name
        self.state_size = np.prod(env.observation_space.shape)
        self.action_size = np.prod(env.action_space.shape)
        self.i = 0
        self.buffer = np.zeros(
            (self.size, self.state_size*2+self.action_size+2), # (s, a, r, s', d)
            dtype=np.float32
        )

    def store(self, state, action, reward, next_state, terminal):
        i = self.i % self.size
        self.buffer[i,:] = np.hstack(
            [state.flatten(), action.flatten(), reward, next_state.flatten(), terminal]
        )
        self.i += 1

    def fill(self):
        print("Filling up buffer")
        state = self.env.reset()
        for _ in trange(self.size-self.i):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.store(state, action, reward, next_state, done)
            if done:
                state = self.env.reset()
            else:
                state = next_state
        print("Buffer filled")
        self.save_to_file()
        
    def full(self):
        return self.i >= self.size

    def sample(self, sample_size):
        sample = self.buffer[
            np.random.choice(self.size, sample_size, replace=False), :]
        return (sample[:, 0:self.state_size],                                                        # state
                sample[:, self.state_size:self.state_size+self.action_size],                         # action
                sample[:, self.state_size+self.action_size],                                         # reward
                sample[:, self.state_size+self.action_size+1: 2*self.state_size+self.action_size+1], # next state
                sample[:, -1])                                                                       # done

    def save_to_file(self):
        with open(f'buffers/{self.env_id}-{self.name}.pkl', 'wb') as f:
            pickle.dump(self, f)
