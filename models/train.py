import os, random, time
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from pong_env import PongEnv
from dqn_agent import DQNAgent

NUM_EPISODES        = 3000
MAX_EPISODE_STEPS   = 3000
BUFFER_SIZE         = 100_000
BATCH_SIZE          = 64
GAMMA               = 0.99
TAU                 = 1e-3
LR                  = 1e-4
UPDATE_FREQ         = 4
HIDDEN_DIM          = 128
EPS_START           = 1.0
EPS_END             = 0.02
EPS_DECAY_STEPS     = 300_000
START_RANDOM_STEPS  = 5000
TARGET_AVG_RETURN   = 0.5           # hard - 0.8 easy - 0.3 mid - 0.5
PRINT_EVERY         = 50
RENDER_EVERY        = 0
MODEL_SAVE_PATH     = "../pong-medium.pth"
SEED                = 44
METRICS_NPZ         = "log_dqn-mid.npz"
CURVE_PNG           = "dqn_pong_training_curve-mid.png"
LOAD_MODEL_FROM     = None         


def train_pong():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    env = PongEnv()
    agent = DQNAgent(
        env.state_dim, env.action_dim,
        BUFFER_SIZE, BATCH_SIZE, LR, GAMMA, TAU,
        UPDATE_FREQ, HIDDEN_DIM
    )

    if LOAD_MODEL_FROM:
        agent.load(LOAD_MODEL_FROM)
        print(f"Loaded weights from {LOAD_MODEL_FROM}")

    returns_hist, wins_hist, lens_hist, steps_hist, loss_hist = [], [], [], [], []
    win100 = deque(maxlen=100)

    total_steps = 0
    best_avg100 = -float('inf')
    solved_at_steps = None
    t0 = time.time()

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        ep_ret, ep_len = 0.0, 0
        render = RENDER_EVERY and ep % RENDER_EVERY == 0

        for _ in range(MAX_EPISODE_STEPS):
            total_steps += 1; ep_len += 1
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-total_steps / EPS_DECAY_STEPS)

            if total_steps < START_RANDOM_STEPS:
                action = random.randint(0, env.action_dim - 1)
            else:
                action = agent.select_action(state, eps)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)

            state = next_state
            ep_ret += reward

            if render:
                try: env.render()
                except Exception: render = False

            if done: break

        win = 1 if ep_ret > 0 else 0
        win100.append(win)

        returns_hist.append(ep_ret)
        wins_hist.append(win)
        lens_hist.append(ep_len)
        steps_hist.append(total_steps)
        loss_hist.append(agent.get_last_loss() or 0.0)

        avg_return_100 = np.mean(returns_hist[-100:])
        avg_win_100    = np.mean(wins_hist[-100:])
        avg_len_100    = np.mean(lens_hist[-100:])

        if ep % PRINT_EVERY == 0:
            print(f"Ep {ep:5d} | R {ep_ret:+5.2f} | Len {ep_len:4d} | "
                  f"AvgR100 {avg_return_100:+.3f} | Win100 {avg_win_100:5.1%} | "
                  f"Eps {eps:.3f} | Loss {loss_hist[-1]:.4f} | "
                  f"Steps {total_steps:7d} | {time.time()-t0:6.1f}s")

        if len(returns_hist) >= 100 and avg_return_100 > best_avg100:
            best_avg100 = avg_return_100
            agent.save(MODEL_SAVE_PATH)
            print(f"★  New best avgReturn100 = {best_avg100:.3f} → saved to {MODEL_SAVE_PATH}")

        if (solved_at_steps is None
                and len(returns_hist) >= 100
                and avg_return_100 >= TARGET_AVG_RETURN):
            solved_at_steps = total_steps
            print(f"✔ Environment solved! AvgReturn100 ≥ {TARGET_AVG_RETURN} "
                  f"after {solved_at_steps} steps.")

    env.close()

    print("\n===== SUMMARY DQN =====")
    print(f"Best rolling-100 return: {best_avg100:.3f}")
    if solved_at_steps:
        print(f"Solved after {solved_at_steps} env-steps.")
    else:
        print("Threshold not reached.")

    np.savez(METRICS_NPZ,
             returns=np.array(returns_hist),
             wins=np.array(wins_hist),
             lengths=np.array(lens_hist),
             steps=np.array(steps_hist),
             losses=np.array(loss_hist))
    print(f"Metrics saved to {METRICS_NPZ}")

    plt.figure(figsize=(10,5))
    plt.plot(steps_hist, returns_hist, alpha=.4, label="episode return")
    if len(returns_hist) >= 100:
        roll = np.convolve(returns_hist, np.ones(100)/100, 'valid')
        plt.plot(steps_hist[99:], roll, lw=2, label="rolling avg (100)")
    plt.axhline(TARGET_AVG_RETURN, c='r', ls='--', label='target')
    plt.xlabel("environment steps"); plt.ylabel("return")
    plt.title("DQN training — Pong 5-D")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(CURVE_PNG)
    print(f"Curve saved to {CURVE_PNG}")


if __name__ == "__main__":
    train_pong()
