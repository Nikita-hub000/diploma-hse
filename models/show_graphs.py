import numpy as np, matplotlib.pyplot as plt
dqn   = np.load("log_dqn.npz")
rein = np.load("policy/log_reinforce.npz")

plt.figure(figsize=(10,4))
plt.plot(dqn["steps"],   dqn["returns"],   label="DQN ep-return",   alpha=.4)
plt.plot(rein["steps"],  rein["returns"],  label="REINFORCE ep-return", alpha=.4)
# сглаженные кривые
win=100
plt.plot(dqn["steps"][win-1:], np.convolve(dqn["returns"], np.ones(win)/win, 'valid'),
         label="DQN avg100", lw=2)
plt.plot(rein["steps"][win-1:], np.convolve(rein["returns"], np.ones(win)/win, 'valid'),
         label="REIN avg100", lw=2)
plt.xlabel("Env steps"); plt.ylabel("Return"); plt.grid(); plt.legend()
plt.tight_layout(); plt.savefig("compare_return.png")
