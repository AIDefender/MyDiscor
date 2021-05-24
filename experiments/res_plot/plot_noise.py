import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

sns.set()
sns.set_context('paper', font_scale=1.5)

ours_name = "lfiw_tper_linear_k10.0"
EXP = "Walker2d-v2"
# paper reward
# AlGOS = ["sac_full", "discor_full", "discor_lfiw_full", ours_name,]
AlGOS = ["sac_full_noise1", "discor_lfiw_full_noise1", ours_name]
# paper ablation
# AlGOS = ["sac_full", "lfiw_full", "tper_linear", ours_name]
# AlGOS = ["sac_full", "discor_full", ours_name]

# paper reward
colors = {
    'sac_full_noise1': 'blue',
    "discor_lfiw_full_noise1": 'green',
    ours_name: 'red',
}
# paper ablation
# colors = {
    # 'sac_full': 'blue',
    # "lfiw_full": 'green',
    # "tper_linear": 'black',
    # ours_name: 'red',
# }
labels = {
    "discor_lfiw_full_noise1": "RM-Discor",
    "discor_full": "Discor",
    'lfiw_full': "Only On-policy Reweight",
    'tper_linear': "Only Step-based Reweight",
    'sac_full_noise1': 'SAC',
    ours_name: 'RM-TCE'
}
MAX_STEP=5e6
ROLLING_STEP=10
# AlGOS = ["discor_full", "lfiw_sac_full", "sac_full"]
root_path = os.path.join("../../logs/"+EXP)

for algo in AlGOS:
    print(algo)
    file = os.path.join(root_path, "%s-all.txt"%algo)
    with open(file, 'r') as f:
        content = f.readlines()
        all_rewards = []
        seed = -100
        for line in content:
            seed += 100
            # if seed != 300:
                # continue
            line_data = []
            for i in line.split(" "):
                try:
                    line_data.append(eval(i))
                except SyntaxError:
                    print("Warn: syntax err")
            print(len(line_data))
            all_rewards.append(line_data[:399])
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
    plt.fill_between(x, rew_mean - rew_std, rew_mean + rew_std, color = colors[algo], alpha = 0.2)
plt.legend()
plt.title("%s-noise1"%EXP)
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.savefig("Reward-%s-noise1.png"%EXP)
# plt.savefig("Ablation-%s.png"%EXP)
