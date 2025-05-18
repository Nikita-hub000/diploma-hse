import random
import collections
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition."""
        # Ensure elements are stored in a consistent format
        experience = (np.array(state, dtype=np.float32),
                      action,
                      reward,
                      np.array(next_state, dtype=np.float32),
                      float(done))
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """Uniformly sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack numpy arrays efficiently
        return (
            np.vstack(states),
            np.array(actions), # Actions are scalars
            np.array(rewards, dtype=np.float32), # Rewards are scalars
            np.vstack(next_states),
            np.array(dones, dtype=np.float32), # Dones are scalars (0.0 or 1.0)
        )

    def __len__(self):
        return len(self.buffer)