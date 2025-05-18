import time
import torch
import numpy as np
import random

from pong_env import PongEnv
from dqn_agent import DQNAgent

def play_pong(
    model_path: str = "../pong.pth", 
    episodes: int = 10,
    render_delay: float = 0.02, 
    hidden_dim: int = 128,     
    seed: int = None
    ):
    env = PongEnv()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        state, _ = env.reset() 
    else:
         state, _ = env.reset()

    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim 
    )

    if not agent.load(model_path):
        print(f"Could not load model from {model_path}. Exiting.")
        env.close()
        return

    agent.qnet.eval()

    print(f"\nStarting Pong playback for {episodes} episodes...")
    print(f"Loading model: {model_path}")

    total_rewards_list = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset() 
        ep_reward = 0.0
        steps = 0
        max_episode_steps = 3000 

        while steps < max_episode_steps:
            steps += 1
            try:
                env.render()
            except Exception as e:
                print(f"Rendering failed (is pygame installed?): {e}")
                break 

            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state

            time.sleep(render_delay)

            if done:
                break

        total_rewards_list.append(ep_reward)
        print(f"Episode {ep:3d} | Reward: {ep_reward:5.1f} | Steps: {steps}")
        if hasattr(env, 'screen') and env.screen is not None:
            time.sleep(0.5)


    env.close() 
    avg_reward = np.mean(total_rewards_list) if total_rewards_list else 0
    print(f"\nPlayback finished. Average Reward over {episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    play_pong(model_path="../pong.pth", episodes=5, hidden_dim=128, render_delay=0.001)
