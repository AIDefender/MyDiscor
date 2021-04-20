import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
args = parser.parse_args()
path1 = "../logs/Ant-v2/sac_eval_Qloss-seed200-20210416-113401/stats/"
path2 = "../logs/Ant-v2/sac_tper46bk_eval_Qloss-seed200-20210416-113407/stats/"
for timestep in range(50000, 2000000, 50000):
    steps = pd.read_csv(path1+"step_timestep%d.txt"%timestep, delimiter='\n', header=None)
    steps2 = 1000 - pd.read_csv(path2+"step_timestep%d.txt"%timestep, delimiter='\n', header=None)
    Qs = pd.read_csv(path1+"Qpi_loss_timestep%d.txt"%timestep, delimiter='\n', header=None)
    Qs2 = pd.read_csv(path2+"Qpi_loss_timestep%d.txt"%timestep, delimiter='\n', header=None)
    Q_value = pd.read_csv(path1+"Q_value%d.txt"%timestep, delimiter='\n', header=None) 
    # plt.scatter(steps[:128], np.sqrt(Qs), label='uniform')
    # plt.scatter(steps2[:128], np.sqrt(Qs2), label='tper')
    plt.scatter(steps[:128], Q_value, label='uniform')
    # plt.scatter(steps[:128], np.log(Qs))
    # plt.ylim(6, 18)
    plt.title("Data from timestep %d"%timestep)
    plt.legend()
    plt.xlabel("Trajectory Step")
    plt.ylabel("Qk - Qpi")
    plt.savefig("figs/%s_Qpi_loss_timestep%d.png"%(args.name, timestep))
    plt.clf()