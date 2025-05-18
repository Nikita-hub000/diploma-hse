# play.py
import time
import torch
import numpy as np
import random

# Import custom environment and agent
from pong_env import PongEnv
from dqn_agent import DQNAgent

def play_pong(
    model_path: str = "../pong.pth", # Path to the saved DQN model
    episodes: int = 10,
    render_delay: float = 0.02, # Delay between frames for visualization
    hidden_dim: int = 128,      # MUST match the hidden_dim used for training!
    seed: int = None
    ):
    """Plays Pong episodes using a trained DQN agent."""

    # --- Environment and Agent Setup ---
    env = PongEnv()

    # Seeding (optional, for deterministic playback if needed)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Note: Env reset seeding might need adjustment if added to PongEnv
        state, _ = env.reset() # Basic reset for now
    else:
         state, _ = env.reset()

    state_dim = env.state_dim
    action_dim = env.action_dim

    # Initialize agent structure (hyperparameters like lr don't matter for playback)
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim # Pass the correct hidden_dim
    )

    # Load the trained weights
    if not agent.load(model_path):
        print(f"Could not load model from {model_path}. Exiting.")
        env.close()
        return

    agent.qnet.eval() # Set the network to evaluation mode (important!)

    print(f"\nStarting Pong playback for {episodes} episodes...")
    print(f"Loading model: {model_path}")

    total_rewards_list = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset() # Reset for the new episode
        ep_reward = 0.0
        steps = 0
        max_episode_steps = 3000 # Same as in training to prevent infinite loops

        while steps < max_episode_steps:
            steps += 1

            # Render the environment
            try:
                env.render()
            except Exception as e:
                print(f"Rendering failed (is pygame installed?): {e}")
                # Optionally break or continue without rendering
                break # Stop if rendering fails

            # Choose action greedily (epsilon=0 for exploitation)
            action = agent.select_action(state, epsilon=0.0)

            # Perform action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            state = next_state

            time.sleep(render_delay)

            if done:
                break

        total_rewards_list.append(ep_reward)
        print(f"Episode {ep:3d} | Reward: {ep_reward:5.1f} | Steps: {steps}")
        # Add a small pause after an episode ends if rendering
        if hasattr(env, 'screen') and env.screen is not None:
            time.sleep(0.5)


    env.close() # Close the pygame window if it was opened
    avg_reward = np.mean(total_rewards_list) if total_rewards_list else 0
    print(f"\nPlayback finished. Average Reward over {episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    # Make sure the model path and hidden_dim match your trained model
    play_pong(model_path="../pong.pth", episodes=5, hidden_dim=128, render_delay=0.001)
