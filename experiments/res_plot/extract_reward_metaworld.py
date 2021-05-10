import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow.compat.v1 as tf
import numpy as np
sns.set_style()

def get_rew_from_path(path):
    event_file = os.listdir(os.path.join(root_path, path, "summary"))[0]
    event_path = os.path.join(root_path, path, "summary", event_file)
    rewards = []
    for event in tf.train.summary_iterator(event_path):
        for value in event.summary.value:
            if value.tag == "reward/test":
                rewards.append(value.simple_value)
    return rewards

AlGOS = ["discor_lfiw_full"]
# AlGOS = ["discor_full", "lfiw_full", "sac_full", "discor_lfiw_full"]
# AlGOS = ["lfiw_tper_linear"]

# for EXP in ["hammer-v1", "push-wall-v1", "dial-turn-v1"]:
for EXP in ["faucet-close-v1"]:
# for EXP in ["button-press-v1", ]:
    # root_path = os.path.join("../../logs/"+EXP)
    root_path = os.path.join("../../../data/discor/logs/"+EXP)
    algo_paths = {}
    for algo in AlGOS:
        algo_paths.update({algo:[]})

    for dir in os.listdir(root_path):
        if ("full" in dir or "linear" in dir) and "txt" not in dir:
            for algo in AlGOS:
                if dir.startswith(algo):
                    algo_paths[algo].append(dir)
    print(EXP)
    print(algo_paths)
    for algo in AlGOS:
        paths = algo_paths[algo]
        file = os.path.join(root_path, "%s-all.txt"%algo)
        with open(file, 'w') as f:
            for path in paths:
                single_seed_reward = get_rew_from_path(path)
                for reward in single_seed_reward:
                    f.write("%.1f "%reward)
                f.write("\n")
