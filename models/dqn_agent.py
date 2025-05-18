import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy # Needed for target network deep copy
from replay_buffer import ReplayBuffer
from model import QNetwork


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        lr: float = 5e-4, 
        gamma: float = 0.99, 
        tau: float = 1e-3, 
        update_freq: int = 4, 
        hidden_dim: int = 128 
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Q-networks (using DDQN logic) - Pass hidden_dim
        self.qnet = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.qnet.state_dict()) # Initial sync
        self.target_net.eval() # Target network is only for inference
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        self.step_count = 0
        self.last_loss = None # To store loss for monitoring

    def select_action(self, state, epsilon: float):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            # Ensure state is a numpy array before converting
            if not isinstance(state, np.ndarray):
                 state = np.array(state, dtype=np.float32)

            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.qnet.eval() # Set to evaluation mode for inference
            with torch.no_grad():
                qvals = self.qnet(state)
            self.qnet.train() # Set back to training mode
            return qvals.argmax().item()

    def step(self, state, action, reward, next_state, done):
        """Add experience and learn if conditions met."""
        # Ensure state and next_state are NumPy arrays or compatible types
        self.buffer.push(np.array(state, dtype=np.float32),
                         action,
                         reward,
                         np.array(next_state, dtype=np.float32),
                         done)
        self.step_count += 1

        # Learn every update_freq steps if buffer has enough samples
        if self.step_count % self.update_freq == 0 and len(self.buffer) >= self.batch_size:
            self.learn()


    def learn(self):
        """Update Q-network using a batch from the replay buffer (DDQN)."""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        # Ensure actions are LongTensor for gather() and have the correct shape [batch_size, 1]
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # Ensure rewards and dones have shape [batch_size, 1]
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device) # Use float for multiplication

        # --- DDQN Target Calculation ---
        with torch.no_grad():
            # 1. Select best actions using the online Q-network for next_states
            online_next_actions = self.qnet(next_states).argmax(1, keepdim=True) # [batch_size, 1]
            # 2. Evaluate these actions using the target Q-network
            target_next_q = self.target_net(next_states).gather(1, online_next_actions) # [batch_size, 1]
            # 3. Compute the TD target
            q_targets = rewards + (self.gamma * target_next_q * (1 - dones)) # dones is 0 or 1
        # --- End DDQN ---

        # Compute current Q estimates for the actions actually taken
        q_values = self.qnet(states).gather(1, actions) # [batch_size, 1]

        # Loss & backprop
        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.last_loss = loss.item() # Store loss

        # --- Soft update target network ---
        self._soft_update_target_network()

    def _soft_update_target_network(self):
        """Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, qnet_param in zip(self.target_net.parameters(), self.qnet.parameters()):
            target_param.data.copy_(self.tau * qnet_param.data + (1.0 - self.tau) * target_param.data)

    def get_last_loss(self):
        """Returns the loss from the last learning step."""
        return self.last_loss

    def save(self, filename):
        """Saves the Q-network weights."""
        torch.save(self.qnet.state_dict(), filename)
        print(f"--- Agent Q-network saved to {filename} ---")

    def load(self, filename):
        """Loads the Q-network weights."""
        try:
            self.qnet.load_state_dict(torch.load(filename, map_location=self.device))
            self.target_net.load_state_dict(self.qnet.state_dict()) # Sync target net
            self.qnet.eval() # Set to eval mode after loading
            self.target_net.eval()
            print(f"--- Agent Q-network loaded from {filename} ---")
            return True
        except FileNotFoundError:
            print(f"--- Error: Agent file not found at {filename} ---")
            return False
        except Exception as e:
             print(f"--- Error loading agent Q-network: {e} ---")
             return False
