import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
sns.set()
sns.set_context('paper', font_scale=1.5)

ENVS = [
    "push-v1",
    "stick-pull-v1",
    "sweep-v1",
    "pick-place-v1",
]
horizons = [
    200,
    320,
    200,
    200
]

ALGOS = [
        "sac_full", 
        # "lfiw_full", 
        "discor_full", 
        "discor_lfiw_full", 
        # "lfiw_tper_linear_full"
        ]

colors = {
    "discor_full": 'green',
    "discor_lfiw_full": 'red',
    "lfiw_full": 'yellow',
    'sac_full': 'blue',
    'lfiw_tper_linear': 'black',
    # 'lfiw_tper_adapt-linear': 'red',
    'lfiw_tper_linear': 'red',
}
labels = {
    "discor_lfiw_full": "ME-Discor",
    "discor_full": "Discor",
    'lfiw_full': "lfiw",
    'sac_full': 'SAC',
    'lfiw_tper_linear': 'lfiw+tper-linear(ours)',
    'lfiw_tper_linear': 'ME-TCE'
}
ROLLING_STEP=10
MAX_STEP=3e6

fig, axs = plt.subplots(1, 4)
index = -1
for EXP, horizon in zip(ENVS, horizons):
    print(EXP)
    index += 1
    this_ax = axs[index]
# for EXP in ["stick-pull-v1", "hammer-v1", "push-wall-v1", "dial-turn-v1"]:
# for EXP in ["hammer-v1", "push-wall-v1", "dial-turn-v1"]:
    # AlGOS = ["discor_full", "lfiw_sac_full", "sac_full"]
    root_path = os.path.join("../../../data/discor/logs/"+EXP)
    # root_path = os.path.join("../../logs/"+EXP)

    for algo in ALGOS:
        print(algo)
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
                all_rewards.append(line_data[:horizon])
        all_rewards = np.array(all_rewards)
        print(all_rewards.shape)
        rew_mean = np.mean(all_rewards, axis=0)
        df = pd.DataFrame(rew_mean)
        rew_mean = df[0].rolling(ROLLING_STEP).mean()
        rew_std = np.std(all_rewards, axis=0)
        x = np.arange(0, MAX_STEP, 5e3)[:len(rew_mean)]
        plot_index = np.arange(0, len(x), 1)
        rew_mean = rew_mean[plot_index]
        rew_std = rew_std[plot_index]
        x = x[plot_index]
        this_ax.plot(x, rew_mean, color=colors[algo], label=labels[algo])
        this_ax.fill_between(x, rew_mean - 0.6*rew_std, rew_mean + 0.6*rew_std, color = colors[algo], alpha = 0.15)
    this_ax.legend()
    this_ax.set_title(EXP)
    this_ax.set_xlabel("Timestep")
    this_ax.set_ylabel("Reward")
    this_ax.ticklabel_format(axis='x', style='sci', scilimits=(4,4))
    this_ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

length=15
fig.set_size_inches(length*3.5, length)
fig.savefig("reward-four-1.png", bbox_inches = 'tight',
    pad_inches = 0.1)
