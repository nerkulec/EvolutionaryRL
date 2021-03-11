import numpy as np
from tqdm.auto import tqdm, trange
import pickle
import pathlib

def get_buffer(size, env, fresh = False, name='1'):
    if not fresh:
        try:
            file_name = f'buffers/{env.spec.id}.pkl'
            with open(file_name, 'rb') as f:
                print('Buffer loaded from disk')
                return pickle.load(f)

        except FileNotFoundError:
            fresh = True
    
    if fresh:
        print('Fresh buffer created')
        return Buffer(size, env, name)

class Buffer:
    def __init__(self, size, env, name='1'):
        self.size = size
        self.env_id = env.spec.id
        self.name = name
        self.state_size = int(np.prod(env.observation_space.shape))
        self.action_size = int(np.prod(env.action_space.shape))
        self.i = 0
        self.buffer = np.zeros(
            (self.size, self.state_size*2+self.action_size+2), # (s, a, r, s', d)
            dtype=np.float32
        )

    def store(self, state, action, reward, next_state, terminal):
        i = self.i % self.size
        
        if type(state) is not int:
            state = state.flatten()
        if type(action) is not int:
            action = action.flatten()
        if type(next_state) is not int:
            next_state = next_state.flatten()

        self.buffer[i,:] = np.hstack(
            [state, action, reward, next_state, terminal]
        )
        self.i += 1

    def fill(self, env):
        print("Filling up buffer")
        state = env.reset()
        for _ in trange(self.size-self.i):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            self.store(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = env.reset()
        print("Buffer filled")
        
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
        directory = 'buffers'
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        with open(f'{directory}/{self.env_id}.pkl', 'wb') as f:
            pickle.dump(self, f)
