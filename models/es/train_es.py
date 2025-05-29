import time, random, os
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from pong_env import PongEnv
from es_agent  import ESAgent

SAVE_PATH   = "../../pong_es-hard.pth"
METRICS_NPZ = "log_es.npz"
CURVE_PNG   = "es_training_curve-hard.png"

EASY_STEP   = 30000000  
MEDIUM_STEP = 50000000 
HARD_STEP   = 75000000  
MAX_EP_LEN = 5000  

SAVE_EASY   = "../../pong_es-easy.pth"
SAVE_MEDIUM = "../../pong_es-medium.pth"
SAVE_HARD   = "../../pong_es-hard.pth"   

def evaluate(agent: ESAgent, episodes=1):
    env = PongEnv()
    total_r = total_len = total_win = 0

    with torch.no_grad():                         
        for _ in range(episodes):
            s,_ = env.reset(); done = False; ep_r = 0; ep_len = 0
            while not done and ep_len < MAX_EP_LEN:
                a = agent.select_action(s)
                s, r, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_r  += r
                ep_len+= 1
            if ep_len >= MAX_EP_LEN and ep_r == 0:
                ep_r = -1
            total_r   += ep_r
            total_len += ep_len
            if ep_r > 0: total_win += 1
    env.close()
    k = episodes
    return total_r/k, total_len/k, total_win/k

def train_es(
    iterations=4000,    
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
    wins_hist, lens_hist = [], []
    best_ret, best_theta     = -1e9, theta.clone()
    env_steps_total          = 0
    ckpt = {"easy":False,"medium":False,"hard":False}
    t0 = time.time()

    for it in range(1, iterations+1):
        noise = torch.randn(pop_size, num_params)
        rewards = torch.empty(pop_size)

        ep_lens = torch.empty(pop_size, dtype=torch.int)
        for k in range(pop_size):
            agent.set_flat_params(theta + sigma * noise[k])
            r, ep_len, _ = evaluate(agent, episodes=1)        
            rewards[k] = r
            ep_lens[k] = ep_len

        env_steps_total += ep_lens.sum().item()

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        grad = (noise.t() @ rewards) / (pop_size * sigma)
        theta += lr * grad

        agent.set_flat_params(theta)
        test_return, test_len, test_win = evaluate(agent, eval_eps)
        returns_hist.append(test_return)
        steps_hist.append(env_steps_total)
        wins_hist.append(test_win)
        lens_hist.append(test_len)

        if not ckpt["easy"] and env_steps_total >= EASY_STEP:
            agent.save(SAVE_EASY)
            ckpt["easy"] = True
        if not ckpt["medium"] and env_steps_total >= MEDIUM_STEP:
            agent.save(SAVE_MEDIUM)
            ckpt["medium"] = True
        if (not ckpt["hard"] and env_steps_total >= HARD_STEP) or (test_return > best_ret):
            best_ret, best_theta = test_return, theta.clone()
            agent.save(SAVE_HARD)
            ckpt["hard"] = True

        if it % 10 == 0:
            print(f"Gen {it:4d} | TestR {test_return:+.2f} | "
                  f"Best {best_ret:+.2f} | Steps {env_steps_total:7d} | "
                  f"{time.time()-t0:5.1f}s")

    print(f"Finished. Best test return {best_ret:+.2f}. Saved to {SAVE_HARD}")
    np.savez(METRICS_NPZ,
             returns=np.array(returns_hist),
             wins=np.array(wins_hist),
             lengths=np.array(lens_hist),
             steps=np.array(steps_hist))
    plt.figure(figsize=(9,4))
    plt.plot(steps_hist, returns_hist)
    plt.xlabel("env steps (approx)"); plt.ylabel("return")
    plt.title("ES training — Pong 5-D"); plt.grid(True); plt.tight_layout()
    plt.savefig(CURVE_PNG)

if __name__ == "__main__":
    train_es()
