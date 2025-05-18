import random, time, os
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from pong_env import PongEnv
from reinforce_agent import ReinforceAgent

EPISODES          = 5000
MAX_STEPS_PER_EP  = 3000
GAMMA             = 0.99
LR                = 1e-3
HIDDEN_DIM        = 128
TARGET_AVG_RETURN = 0.8      
PRINT_EVERY       = 50
RENDER_EVERY      = 0          
SAVE_MODEL_PATH   = "../pong_reinforce.pth"
SEED              = 44
METRICS_NPZ       = "log_reinforce.npz"
CURVE_PNG         = "reinforce_pong_curve.png"

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    env   = PongEnv()
    agent = ReinforceAgent(env.state_dim, env.action_dim,
                           gamma=GAMMA, lr=LR, hidden_dim=HIDDEN_DIM)

    returns_hist, wins_hist, lens_hist, steps_hist, loss_hist = [], [], [], [], []
    win100 = deque(maxlen=100)

    env_steps_total = 0
    solved_at_steps = None
    best_avg100     = -float('inf')
    t0 = time.time()

    for ep in range(1, EPISODES + 1):

        state, _ = env.reset()
        ep_ret   = 0.0
        render   = RENDER_EVERY and ep % RENDER_EVERY == 0

        for step in range(MAX_STEPS_PER_EP):

            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.step(reward, done)

            state   = next_state
            ep_ret += reward
            env_steps_total += 1

            if render:
                try: env.render()
                except Exception: render = False

            if done:
                break

        ep_len  = step + 1
        win     = 1 if ep_ret > 0 else 0
        win100.append(win)

        returns_hist.append(ep_ret)
        wins_hist.append(win)
        lens_hist.append(ep_len)
        steps_hist.append(env_steps_total)
        loss_hist.append(agent.get_last_loss() or 0.0)

        avg_return_100 = np.mean(returns_hist[-100:])
        avg_win_100    = np.mean(wins_hist[-100:])
        avg_len_100    = np.mean(lens_hist[-100:])

        if ep % PRINT_EVERY == 0:
            print(f"Ep {ep:5d} | "
                  f"R {ep_ret:+5.2f} | Len {ep_len:4d} | "
                  f"AvgR100 {avg_return_100:+.3f} | Win100 {avg_win_100:5.1%} | "
                  f"Loss {loss_hist[-1]:.4f} | Steps {env_steps_total:7d} | "
                  f"{time.time()-t0:6.1f}s")

        if len(returns_hist) >= 100 and avg_return_100 > best_avg100:
            best_avg100 = avg_return_100
            agent.save(SAVE_MODEL_PATH)
            print(f"★  New best avgReturn100 = {best_avg100:.3f} → saved to {SAVE_MODEL_PATH}")

        if (solved_at_steps is None
                and len(returns_hist) >= 100
                and avg_return_100 >= TARGET_AVG_RETURN):
            solved_at_steps = env_steps_total
            print(f"✔ Environment solved! AvgReturn100 ≥ {TARGET_AVG_RETURN} "
                  f"after {solved_at_steps} steps.")

    env.close()

    print("\n===== SUMMARY =====")
    print(f"Best rolling-100 return: {best_avg100:.3f}")
    if solved_at_steps:
        print(f"Solved after {solved_at_steps} environment steps.")
    else:
        print("Threshold not reached.")

    np.savez(METRICS_NPZ,
             returns=np.array(returns_hist),
             wins=np.array(wins_hist),
             lengths=np.array(lens_hist),
             steps=np.array(steps_hist),
             losses=np.array(loss_hist))
    print(f"Metrics saved to {METRICS_NPZ}")

    plt.figure(figsize=(10, 5))
    plt.plot(steps_hist, returns_hist, alpha=0.4, label="episode return")
    if len(returns_hist) >= 100:
        roll = np.convolve(returns_hist, np.ones(100)/100, mode='valid')
        plt.plot(steps_hist[99:], roll, lw=2, label="rolling avg (100)")
    plt.axhline(TARGET_AVG_RETURN, color='r', ls='--', label='target')
    plt.xlabel("environment steps")
    plt.ylabel("return")
    plt.title("REINFORCE training — Pong 5-D")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(CURVE_PNG)
    print(f"Curve saved to {CURVE_PNG}")


if __name__ == "__main__":
    main()
