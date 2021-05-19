import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import polyfit
import argparse
import seaborn as sns
sns.set()
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--env", type=str)
parser.add_argument("--bound", type=int, nargs="*", default=None)
args = parser.parse_args()
# path1 = "../../logs/Ant-v2/eval_sac_bkwardstepr_ori_Qpi-seed0-20210421-114746/stats/"
# path1 = "../../logs/Hopper-v2/eval_sac_bkwardstepr_ori_Qpi-seed0-20210424-215547/stats/"
all_paths = {
    "Humanoid": [
        "../../logs/Humanoid-v2/eval_sac_bkwardstepr_ori_Qpi-seed0-20210423-185908/stats/",
        # "../../logs/Humanoid-v2/eval_discor_bkwardstepr_ori_Qpi-seed0-20210424-091355/stats/",
        "../../logs/Humanoid-v2/eval_tper_bkwardstepr_ori_Qpi-seed0-20210424-091248/stats/",
        "../../logs/Humanoid-v2/eval_tper_linear1.5_0.6_3.0_-0.3-seed0-20210426-220241/stats/"
    ],
    "Hopper": [
        "../../logs/Hopper-v2/eval_sac_bkwardstepr_ori_Qpi_test_donecnt-seed0-20210425-164242/stats/",
        "../../logs/Hopper-v2/eval_discor_bkwardstepr_ori_Qpi-seed0-20210424-215522/stats/",
        "../../logs/Hopper-v2/eval_tper-seed0-20210425-214013/stats/"
    ],
    "Ant": [
        "../../logs/Ant-v2/eval_sac_bkwardstepr_ori_Qpi-seed0-20210421-114746/stats/",
        # "../../logs/Ant-v2/eval_discor_bkwardstepr_ori_Qpi-seed0-20210420-100559/stats/",
        # "../../logs/Ant-v2/eval_tper_bkwardstepr_ori_Qpi-seed0-20210421-114717/stats/",
        # "../../logs/Ant-v2/eval_tper_linear1.5_0.6_3.0_-0.3-seed200-20210426-220133/stats/",
    ],
    "Walker": [
        # "../../logs/Walker2d-v2/eval_sac_bkwardstep-seed0-20210418-140650/stats/",
        "../../logs/Walker2d-v2/eval_discor_bkwardstep_ori_Qpi-seed0-20210419-115020/stats/",
        # "../../logs/Walker2d-v2/eval_tper_bkwardstep_ori_Qpi-seed0-20210425-214121/stats/"
    ]
}
paths = all_paths["Walker"]
labels = [
    "sac",
    # "discor",
    # "tper",
    # "tper_linear"
]
env = paths[0].split("/")[3]
xaxis = 'step'
type = 'Qloss'

def plot_single_path(path, xaxis='step', yaxis='Qloss', prefix='uniform', scatter_depth=1.):
    xdata = None
    if xaxis == 'step':
        steps = pd.read_csv(path+"step_timestep%d.txt"%timestep, delimiter='\n', header=None)
        xdata = steps
    else:
        done_cnt = pd.read_csv(path+"done_cnt_timestep%d.txt"%timestep, delimiter='\n', header=None)
        xdata = done_cnt
    
    Qpi = pd.read_csv(path+"Qpi_timestep%d.txt"%timestep, delimiter='\n', header=None)
    Qpred = pd.read_csv(path+"Qvalue_timestep%d.txt"%timestep, delimiter='\n', header=None) 
    ydata = None
    if yaxis == 'Qloss':
        Qloss = Qpi-Qpred
        # Qloss = np.abs(Qpi-Qpred)
        ydata = Qloss
    elif yaxis == 'Qpi':
        ydata = Qpi 
    elif yaxis == 'Qpred':
        ydata = Qpred
    elif yaxis == 'err_pred':
        err_pred = pd.read_csv(path+"Error_pred_timestep%d.txt"%timestep, delimiter='\n', header=None)
        ydata = err_pred
        # plot real Q loss for comparison
        Qloss = np.abs(Qpi-Qpred)
        plt.scatter(xdata, Qloss, label='real Q loss', alpha=scatter_depth)
    plt.scatter(xdata, ydata, alpha=scatter_depth)
    coeff = polyfit(np.squeeze(xdata), np.squeeze(ydata), 2)
    x = np.arange(int(xdata.min()), int(xdata.max()), 5)
    # plt.plot(x, coeff[0]*x**2+coeff[1]*x+coeff[2], label=prefix, linewidth=3)

dir_name = "figs/%s/%s-%s"%(env, type, xaxis)
print("Fig name "+dir_name)

for timestep in range(50000, 2000000, 50000):
    if args.bound is not None:
        plt.ylim(args.bound[0], args.bound[1])
    for path, label in zip(paths, labels):
        plot_single_path(path, xaxis, type, label, 1)
    plt.title("Data from timestep %d in env %s"%(timestep, env))
    # plt.legend()
    plt.xlabel(xaxis)
    plt.ylabel("Qreal - Qpred")
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    plt.savefig("figs/%s/%s-%s/%d.png"%(env, type, xaxis, timestep))
    plt.clf()