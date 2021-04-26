import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
sns.set()

AlGOS = ["discor_full", "sac_full", "lfiw_full", "discor_lfiw_full"]
# AlGOS = ["discor_full", "sac_full", "lfiw_sac_full", ]
# AlGOS = ["discor_full", "sac_full", "lfiw_full", "lfiw_discor_full"]
colors = {
    AlGOS[0]: 'red',
    AlGOS[1]: 'blue',
    AlGOS[2]: 'yellow',
    AlGOS[3]: 'black',
}
labels = {
    AlGOS[0]: "discor",
    AlGOS[1]: 'sac',
    AlGOS[2]: 'lfiw',
    AlGOS[3]: 'discor_lfiw(ours)',
}
ROLLING_STEP=10
MAX_STEP=3e6
# for EXP in ["stick-pull-v1"]:
for EXP in ["door-open-v1", ]:
# for EXP in ["stick-pull-v1", "hammer-v1", "push-wall-v1", "dial-turn-v1"]:
# for EXP in ["hammer-v1", "push-wall-v1", "dial-turn-v1"]:
    # AlGOS = ["discor_full", "lfiw_sac_full", "sac_full"]
    root_path = os.path.join("../../../data/discor/logs/"+EXP)
    # root_path = os.path.join("../../logs/"+EXP)

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
                all_rewards.append(line_data[:267])
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
        plt.plot(x, rew_mean, color=colors[algo], label=labels[algo])
        plt.fill_between(x, rew_mean - 0.6*rew_std, rew_mean + 0.6*rew_std, lw = 3, color = colors[algo], alpha = 0.2)
    plt.legend()
    plt.title(EXP)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(4,4))
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.savefig("reward-%s.png"%EXP)
    plt.clf()
