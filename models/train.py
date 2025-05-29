"""
train_dqn.py — версия с автоматическим созданием трёх уровней сложности:
• 500 000 шагов  → easy
• 1 000 000 шагов → medium
• 1 900 000 шагов (или конец обучения) → hard
Каждый чек-пойнт сразу сохраняется и позже может быть конвертирован в ONNX.
"""

import os, random, time
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from pong_env import PongEnv
from dqn_agent import DQNAgent

# ─── Гиперпараметры обучения ──────────────────────────────────────
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

# пороги, на которых фиксируем уровни сложности
EASY_STEP           = 500_000
MEDIUM_STEP         = 1_000_000
HARD_STEP           = 1_900_000   # можно править

TARGET_AVG_RETURN   = 0.5
PRINT_EVERY         = 50
RENDER_EVERY        = 0
SEED                = 44

# пути для сохранения
SAVE_DIR            = "../dqn"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_EASY           = os.path.join(SAVE_DIR, "dqn_easy.pth")
SAVE_MEDIUM         = os.path.join(SAVE_DIR, "dqn_medium.pth")
SAVE_HARD           = os.path.join(SAVE_DIR, "dqn_hard.pth")

METRICS_NPZ         = "log_dqn.npz"
CURVE_PNG           = "dqn_curve.png"
LOAD_MODEL_FROM     = None
# ──────────────────────────────────────────────────────────────────


def train_pong() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    env = PongEnv()
    agent = DQNAgent(
        env.state_dim, env.action_dim, BUFFER_SIZE, BATCH_SIZE,
        LR, GAMMA, TAU, UPDATE_FREQ, HIDDEN_DIM
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

    ckpt_flags = {"easy": False, "medium": False, "hard": False}

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

            # визуализация
            if render:
                try: env.render()
                except Exception:
                    render = False

            if done:
                break

        # ── накопление метрик ───────────────────
        win = 1 if ep_ret > 0 else 0
        win100.append(win)

        returns_hist.append(ep_ret)
        wins_hist.append(win)
        lens_hist.append(ep_len)
        steps_hist.append(total_steps)
        loss_hist.append(agent.get_last_loss() or 0.0)

        avg_return_100 = np.mean(returns_hist[-100:])

        # ── вывод в консоль ─────────────────────
        if ep % PRINT_EVERY == 0:
            avg_win_100 = np.mean(wins_hist[-100:])
            print(f"Ep {ep:5d} | R {ep_ret:+5.2f} | Len {ep_len:4d} | "
                  f"AvgR100 {avg_return_100:+.3f} | Win100 {avg_win_100:5.1%} | "
                  f"Eps {eps:.3f} | Loss {loss_hist[-1]:.4f} | "
                  f"Steps {total_steps:7d} | {time.time()-t0:6.1f}s")

        # ── периодическое сохранение «лучшей» модели ──────────
        if len(returns_hist) >= 100 and avg_return_100 > best_avg100:
            best_avg100 = avg_return_100
            agent.save(SAVE_HARD)   # hard перезаписывается «самым лучшим»
            print(f"★ New best avgR100 = {best_avg100:.3f} → saved to {SAVE_HARD}")

        # ── чек-пойнты сложности ───────────────────────────────
        if not ckpt_flags["easy"] and total_steps >= EASY_STEP:
            agent.save(SAVE_EASY); ckpt_flags["easy"] = True
            print(f"[CKPT] easy-модель сохранена на {EASY_STEP:,} шагах → {SAVE_EASY}")

        if not ckpt_flags["medium"] and total_steps >= MEDIUM_STEP:
            agent.save(SAVE_MEDIUM); ckpt_flags["medium"] = True
            print(f"[CKPT] medium-модель сохранена на {MEDIUM_STEP:,} шагах → {SAVE_MEDIUM}")

        if not ckpt_flags["hard"] and total_steps >= HARD_STEP:
            agent.save(SAVE_HARD); ckpt_flags["hard"] = True
            print(f"[CKPT] hard-модель сохранена на {HARD_STEP:,} шагах → {SAVE_HARD}")

        # ── условие «решено» ───────────────────────────────────
        if (solved_at_steps is None and len(returns_hist) >= 100
                and avg_return_100 >= TARGET_AVG_RETURN):
            solved_at_steps = total_steps
            print(f"✔ Environment solved! AvgR100 ≥ {TARGET_AVG_RETURN} "
                  f"after {solved_at_steps} steps.")

    env.close()

    # ── сохранение метрик и кривой ─────────────────────────────
    np.savez(METRICS_NPZ,
             returns=np.array(returns_hist),
             wins=np.array(wins_hist),
             lengths=np.array(lens_hist),
             steps=np.array(steps_hist),
             losses=np.array(loss_hist))
    print(f"Metrics saved → {METRICS_NPZ}")

    plt.figure(figsize=(10, 5))
    plt.plot(steps_hist, returns_hist, alpha=0.4, label="episode return")
    if len(returns_hist) >= 100:
        roll = np.convolve(returns_hist, np.ones(100)/100, 'valid')
        plt.plot(steps_hist[99:], roll, lw=2, label="rolling avg (100)")
    plt.axhline(TARGET_AVG_RETURN, c='r', ls='--', label='target')
    plt.xlabel("environment steps"); plt.ylabel("return")
    plt.title("DQN training — Pong 5-D")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(CURVE_PNG)
    print(f"Curve saved → {CURVE_PNG}")


if __name__ == "__main__":
    train_pong()
