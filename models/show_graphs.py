import numpy as np, matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 120

def load_log(path, prefix):
    d = np.load(path); return {f"{prefix}_{k}": d[k] for k in d.files}

dqn  = load_log("log_dqn.npz",              "dqn")
pg   = load_log("policy/log_reinforce.npz", "pg")
es   = load_log("es/log_es.npz",            "es")

def rolling(arr, w=100):
    return np.convolve(arr, np.ones(w)/w, 'valid') if len(arr) >= w else arr*0

# --- ограничиваем ES тем же горизонтом, что у DQN/PG -------------
max_common = max(dqn["dqn_steps"][-1], pg["pg_steps"][-1])

# получаем булеву маску ровно под массив шагов
mask_es = es["es_steps"] <= max_common
es_steps_short = es["es_steps"][mask_es]                # укороченный steps
lim = len(es_steps_short)                               # целевая длина

# обрезаем все остальные массивы ES до той же длины
for key in list(es.keys()):
    if key.endswith("_steps"):
        es[key] = es_steps_short
    else:
        es[key] = es[key][:lim]

# -------------------------- Графики -------------------------------
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 1) Return
for tag, color in [("dqn", "tab:blue"), ("pg", "tab:orange"), ("es", "tab:green")]:
    s   = globals()[tag]["{}_steps".format(tag)]
    ret = globals()[tag]["{}_returns".format(tag)]
    ax[0].plot(s, ret, alpha=.25, color=color)
    ax[0].plot(s[99:], rolling(ret), lw=2, color=color, label=tag.upper())
ax[0].set_ylabel("Return"); ax[0].grid(True); ax[0].legend()

# 2) Win-Rate
for tag, color in [("dqn", "tab:blue"), ("pg", "tab:orange"), ("es", "tab:green")]:
    s   = globals()[tag]["{}_steps".format(tag)]
    win = globals()[tag]["{}_wins".format(tag)]
    ax[1].plot(s[99:], rolling(win), lw=2, color=color, label=tag.upper())
ax[1].set_ylabel("Win-Rate"); ax[1].grid(True); ax[1].legend()

# 3) Length
for tag, color in [("dqn", "tab:blue"), ("pg", "tab:orange"), ("es", "tab:green")]:
    s   = globals()[tag]["{}_steps".format(tag)]
    ln  = globals()[tag]["{}_lengths".format(tag)]
    ax[2].plot(s[99:], rolling(ln), lw=2, color=color, label=tag.upper())
ax[2].set_ylabel("Length"); ax[2].set_xlabel("Env steps"); ax[2].grid(True); ax[2].legend()

ax[2].set_xlim(0, max_common)          # одинаковый диапазон оси X
fig.suptitle("Pong: сравнение DQN / PG / ES (общий горизонтом)")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig("compare_all_metrics_clipped.png")
print("Saved → compare_all_metrics_clipped.png")
