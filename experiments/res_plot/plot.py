import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
sns.set()

EXP = "Ant-v2"
AlGOS = ["lfiw_sac_full", "sac_full"]
colors = {
    "lfiw_sac_full": 'red',
    'sac_full': 'blue'
}
labels = {
    'lfiw_sac_full': "lfiw_sac",
    'sac_full': 'sac'
}
MAX_STEP=4e6
# AlGOS = ["discor_full", "lfiw_sac_full", "sac_full"]
root_path = os.path.join("../../logs/"+EXP)

for algo in AlGOS:
    file = os.path.join(root_path, "%s-all.txt"%algo)
    with open(file, 'r') as f:
        content = f.readlines()
        all_rewards = []
        for line in content:
            all_rewards.append([eval(i) for i in line.split(" ")[:-1]])
        all_rewards = np.array(all_rewards)
    rew_mean = np.mean(all_rewards, axis=0)
    rew_std = np.std(all_rewards, axis=0)
    x = np.arange(0, MAX_STEP, 5e3)[:len(rew_mean)]
    plt.plot(x, rew_mean, color=colors[algo], label=labels[algo])
    plt.fill_between(x, rew_mean - rew_std, rew_mean + rew_std, lw = 3, color = colors[algo], alpha = 0.1)
plt.legend()
plt.title(EXP)
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.savefig("reward-%s.png"%EXP)