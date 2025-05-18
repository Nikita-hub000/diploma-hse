import random
import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (np.array(state, dtype=np.float32),
                      action,
                      reward,
                      np.array(next_state, dtype=np.float32),
                      float(done))
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.vstack(states),
            np.array(actions), 
            np.array(rewards, dtype=np.float32), 
            np.vstack(next_states),
            np.array(dones, dtype=np.float32), 
        )

    def __len__(self):
        return len(self.buffer)
