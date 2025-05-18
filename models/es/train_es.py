import time, random, os
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from pong_env import PongEnv
from es_agent  import ESAgent

SAVE_PATH   = "../../pong_es-hard.pth"
METRICS_NPZ = "log_es-hard.npz"
CURVE_PNG   = "es_training_curve-hard.png"

def evaluate(agent: ESAgent, episodes=3) -> float:
    env, score = PongEnv(), 0.0
    for _ in range(episodes):
        state,_ = env.reset(); done=False
        while not done:
            action = agent.select_action(state)
            state,reward,done,_,_ = env.step(action)
            score += reward
    env.close()
    return score / episodes

def train_es(
    iterations=500,    
    pop_size=50,
    sigma=0.1,
    lr=0.02,
    eval_eps=3,
    seed=44,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    agent        = ESAgent()
    theta        = agent.get_flat_params()
    num_params   = theta.numel()

    returns_hist, steps_hist = [], []
    best_ret, best_theta     = -1e9, theta.clone()
    env_steps_total          = 0
    t0 = time.time()

    for it in range(1, iterations+1):
        noise = torch.randn(pop_size, num_params)
        rewards = torch.empty(pop_size)

        for k in range(pop_size):
            agent.set_flat_params(theta + sigma * noise[k])
            r = evaluate(agent, episodes=1)        
            rewards[k] = r
            env_steps_total += 3000               

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        grad = (noise.t() @ rewards) / (pop_size * sigma)
        theta += lr * grad

        agent.set_flat_params(theta)
        test_return = evaluate(agent, eval_eps)
        returns_hist.append(test_return)
        steps_hist.append(env_steps_total)

        if test_return > best_ret:
            best_ret, best_theta = test_return, theta.clone()
            agent.save(SAVE_PATH)

        if it % 10 == 0:
            print(f"Gen {it:4d} | TestR {test_return:+.2f} | "
                  f"Best {best_ret:+.2f} | Steps {env_steps_total:7d} | "
                  f"{time.time()-t0:5.1f}s")

    print(f"Finished. Best test return {best_ret:+.2f}. Saved to {SAVE_PATH}")
    np.savez(METRICS_NPZ, returns=np.array(returns_hist), steps=np.array(steps_hist))
    plt.figure(figsize=(9,4))
    plt.plot(steps_hist, returns_hist)
    plt.xlabel("env steps (approx)"); plt.ylabel("return")
    plt.title("ES training — Pong 5-D"); plt.grid(True); plt.tight_layout()
    plt.savefig(CURVE_PNG)

if __name__ == "__main__":
    train_es()
