import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
sns.set()
sns.set_context('paper', font_scale=1.5)

EXP = "Hopper-v2"
# AlGOS = ["sac_full", "discor_full", "lfiw_full", "discor_lfiw_full"]
AlGOS = ["sac_full", "discor_full", "lfiw_tper_linear_k10.0", 'discor_lfiw_full']
# AlGOS = ["sac_full", "lfiw_full", "lfiw_tper_linear"]
# AlGOS = ["discor_lfiw_full", "lfiw_full", "lfiw_tper_adapt-linear"]
colors = {
    "discor_full": 'green',
    "discor_lfiw_full": 'black',
    "lfiw_full": 'yellow',
    'sac_full': 'blue',
    # 'lfiw_tper_adapt-linear': 'red',
    'lfiw_tper_linear_k10.0': 'red',
}
labels = {
    "discor_lfiw_full": "ME-Discor",
    "discor_full": "Discor",
    'lfiw_full': "lfiw",
    'sac_full': 'SAC',
    'lfiw_tper_linear_k10.0': 'ME-TCE'
}
MAX_STEP=5e6
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
    print(rew_mean.shape)
    df = pd.DataFrame(rew_mean)
    rew_mean = df[0].rolling(ROLLING_STEP).mean()
    rew_std = np.std(all_rewards, axis=0)
    x = np.arange(0, MAX_STEP, 5e3)[:len(rew_mean)]
    plot_index = np.arange(0, len(x), 1)
    rew_mean = rew_mean[plot_index]
    rew_std = rew_std[plot_index]
    rew_std = np.clip(rew_std, 0, 1500)
    x = x[plot_index]
    plt.plot(x, rew_mean, color=colors[algo], label=labels[algo])
    plt.fill_between(x, rew_mean - rew_std, rew_mean + rew_std, lw = 3, color = colors[algo], alpha = 0.1)
plt.legend()
plt.title(EXP)
plt.xlabel("Timestep")
plt.ylabel("Reward")
# plt.savefig("Ablation-%s.png"%EXP)
plt.savefig("Reward-%s.png"%EXP)
