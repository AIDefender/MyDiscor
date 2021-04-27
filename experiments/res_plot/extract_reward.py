import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow.compat.v1 as tf
import numpy as np
sns.set_style()

EXP = "Ant-v2"
# AlGOS = ["lfiw_sac", "sac"]
# AlGOS = ["discor_full", "lfiw_tper_full"]
AlGOS = ["tper_linear"]
root_path = os.path.join("../../logs/"+EXP)

def get_rew_from_path(path):
    event_file = os.listdir(os.path.join(root_path, path, "summary"))[0]
    event_path = os.path.join(root_path, path, "summary", event_file)
    rewards = []
    for event in tf.train.summary_iterator(event_path):
        for value in event.summary.value:
            if value.tag == "reward/test":
                rewards.append(value.simple_value)
    return rewards

algo_paths = {}
for algo in AlGOS:
    algo_paths.update({algo:[]})

for dir in os.listdir(root_path):
    if "full" in dir and "txt" not in dir or "plt" in dir:
        for algo in AlGOS:
            if dir.startswith(algo):
                algo_paths[algo].append(dir)
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