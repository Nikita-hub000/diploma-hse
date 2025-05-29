"""
train_reinforce.py  —  версия с чек-пойнтами сложности
  • 0.5 М шагов  →  easy   (reinforce_easy.pth)
  • 1.0 М шагов  →  medium (reinforce_medium.pth)
  • 1.9 М шагов  →  hard   (reinforce_hard.pth)

Сохраняет те же метрики, что и DQN-скрипт, в формате .npz:
returns, wins, lengths, steps, losses
"""

import os, random, time
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from pong_env import PongEnv
from reinforce_agent import ReinforceAgent

# ── гиперпараметры обучения ──────────────────────────────────────
EPISODES          = 8000
MAX_STEPS_PER_EP  = 3000
GAMMA             = 0.99
LR                = 1e-3
HIDDEN_DIM        = 128

EASY_STEP         = 500_000
MEDIUM_STEP       = 1_000_000
HARD_STEP         = 1_900_000       # или конец обучения

TARGET_AVG_RETURN = 0.80
PRINT_EVERY       = 50
RENDER_EVERY      = 0
SEED              = 44

SAVE_DIR          = "../../policy"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_EASY         = os.path.join(SAVE_DIR, "reinforce_easy.pth")
SAVE_MEDIUM       = os.path.join(SAVE_DIR, "reinforce_medium.pth")
SAVE_HARD         = os.path.join(SAVE_DIR, "reinforce_hard.pth")

METRICS_NPZ       = "log_reinforce.npz"
CURVE_PNG         = "reinforce_curve.png"
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    env   = PongEnv()
    agent = ReinforceAgent(env.state_dim, env.action_dim,
                           gamma=GAMMA, lr=LR, hidden_dim=HIDDEN_DIM)

    returns_hist, wins_hist, lens_hist, steps_hist, loss_hist = [], [], [], [], []
    win100 = deque(maxlen=100)

    env_steps_total = 0
    best_avg100     = -float('inf')
    solved_at_steps = None
    ckpt_flags      = {"easy": False, "medium": False, "hard": False}
    t0 = time.time()

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        ep_ret = 0.0
        render = RENDER_EVERY and ep % RENDER_EVERY == 0

        for step in range(MAX_STEPS_PER_EP):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.step(reward, done)          # внутреннее обновление
            state = next_state
            ep_ret += reward
            env_steps_total += 1

            if render:
                try: env.render()
                except Exception:
                    render = False
            if done:
                break

        ep_len = step + 1
        win    = 1 if ep_ret > 0 else 0
        win100.append(win)

        returns_hist.append(ep_ret)
        wins_hist.append(win)
        lens_hist.append(ep_len)
        steps_hist.append(env_steps_total)
        loss_hist.append(agent.get_last_loss() or 0.0)

        avg_return_100 = np.mean(returns_hist[-100:])

        # --- консольный вывод -----------------------------------
        if ep % PRINT_EVERY == 0:
            avg_win_100 = np.mean(wins_hist[-100:])
            print(f"Ep {ep:5d} | R {ep_ret:+5.2f} | Len {ep_len:4d} | "
                  f"AvgR100 {avg_return_100:+.3f} | Win100 {avg_win_100:5.1%} | "
                  f"Loss {loss_hist[-1]:.4f} | Steps {env_steps_total:7d} | "
                  f"{time.time()-t0:6.1f}s")

        # --- чек-пойнты сложности -------------------------------
        if not ckpt_flags["easy"] and env_steps_total >= EASY_STEP:
            agent.save(SAVE_EASY); ckpt_flags["easy"] = True
            print(f"[CKPT] сохранили easy-уровень на {EASY_STEP:,} шагах → {SAVE_EASY}")

        if not ckpt_flags["medium"] and env_steps_total >= MEDIUM_STEP:
            agent.save(SAVE_MEDIUM); ckpt_flags["medium"] = True
            print(f"[CKPT] сохранили medium-уровень на {MEDIUM_STEP:,} шагах → {SAVE_MEDIUM}")

        if not ckpt_flags["hard"] and env_steps_total >= HARD_STEP:
            agent.save(SAVE_HARD); ckpt_flags["hard"] = True
            print(f"[CKPT] сохранили hard-уровень на {HARD_STEP:,} шагах → {SAVE_HARD}")

        # --- «лучшая» модель (по rolling-100) -------------------
        if len(returns_hist) >= 100 and avg_return_100 > best_avg100:
            best_avg100 = avg_return_100
            agent.save(SAVE_HARD)      # hard-файл переопределяется
            print(f"★ New best avgR100 = {best_avg100:.3f} → saved to {SAVE_HARD}")

        # --- условие «решено» -----------------------------------
        if (solved_at_steps is None and len(returns_hist) >= 100
                and avg_return_100 >= TARGET_AVG_RETURN):
            solved_at_steps = env_steps_total
            print(f"✔ Environment solved! AvgR100 ≥ {TARGET_AVG_RETURN} "
                  f"after {solved_at_steps} steps.")

    env.close()

    # --- сохранение метрик -------------------------------------
    np.savez(METRICS_NPZ,
             returns=np.array(returns_hist),
             wins=np.array(wins_hist),
             lengths=np.array(lens_hist),
             steps=np.array(steps_hist),
             losses=np.array(loss_hist))
    print(f"Metrics saved → {METRICS_NPZ}")

    plt.figure(figsize=(10, 5))
    plt.plot(steps_hist, returns_hist, alpha=.4, label="episode return")
    if len(returns_hist) >= 100:
        roll = np.convolve(returns_hist, np.ones(100)/100, 'valid')
        plt.plot(steps_hist[99:], roll, lw=2, label="rolling avg (100)")
    plt.axhline(TARGET_AVG_RETURN, color='r', ls='--', label='target')
    plt.xlabel("environment steps"); plt.ylabel("return")
    plt.title("REINFORCE training — Pong 5-D")
    plt.grid(); plt.legend(); plt.tight_layout()
    plt.savefig(CURVE_PNG)
    print(f"Curve saved → {CURVE_PNG}")


if __name__ == "__main__":
    main()
