import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
sns.set()

EXP = "Ant-v2"
# AlGOS = ["discor_full", "lfiw_sac_full", "sac_full"]
AlGOS = ["lfiw_sac_full", "sac_full", "discor_full", "lfiw_tper_full", "tper_linear"]
# AlGOS = ["lfiw_sac_full", "sac_full", "discor_full", "lfiw_tper_full"]
colors = {
    "discor_full": 'green',
    "lfiw_sac_full": 'yellow',
    'sac_full': 'blue',
    'lfiw_tper_full': 'black',
    'tper_linear': 'red',
}
labels = {
    "discor_full": "discor",
    'lfiw_sac_full': "lfiw",
    'sac_full': 'sac',
    'lfiw_tper_full': 'lfiw+tper-hard(ours)',
    'tper_linear': 'tper-linear(ours)'
}
MAX_STEP=4e6
ROLLING_STEP=10
# AlGOS = ["discor_full", "lfiw_sac_full", "sac_full"]
root_path = os.path.join("../../logs/"+EXP)

for algo in AlGOS:
    file = os.path.join(root_path, "%s-all.txt"%algo)
    with open(file, 'r') as f:
        content = f.readlines()
        all_rewards = []
        for line in content:
            line_data = []
            for i in line.split(" "):
                try:
                    line_data.append(eval(i))
                except SyntaxError:
                    print("Warn: syntax err")
            print(len(line_data))
            all_rewards.append(line_data[:400])
        all_rewards = np.array(all_rewards)
    rew_mean = np.mean(all_rewards, axis=0)
    df = pd.DataFrame(rew_mean)
    rew_mean = df[0].rolling(ROLLING_STEP).mean()
    rew_std = np.std(all_rewards, axis=0)
    x = np.arange(0, MAX_STEP, 5e3)[:len(rew_mean)]
    plot_index = np.arange(0, len(x), 1)
    rew_mean = rew_mean[plot_index]
    rew_std = rew_std[plot_index]
    x = x[plot_index]
    plt.plot(x, rew_mean, color=colors[algo], label=labels[algo])
    plt.fill_between(x, rew_mean - rew_std, rew_mean + rew_std, lw = 3, color = colors[algo], alpha = 0.1)
plt.legend()
plt.title(EXP)
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.savefig("reward-%s.png"%EXP)