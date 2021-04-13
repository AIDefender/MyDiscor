import pandas as pd
import matplotlib.pyplot as plt
for timestep in range(500, 7000, 500):
    steps = pd.read_csv("../logs/ContinuousGrid-v0/sac_test-seed0-20210413-101218/step_timestep%d.txt"%timestep, delimiter='\n', header=None)
    Qs = pd.read_csv("../logs/ContinuousGrid-v0/sac_test-seed0-20210413-101218/Qpi_loss_timestep%d.txt"%timestep, delimiter='\n', header=None)
    plt.scatter(steps, Qs)
    plt.title("Data from timestep %d"%timestep)
    plt.xlabel("Trajectory Step")
    plt.ylabel("Qk - Qpi")
    plt.savefig("figs/Qpi_loss_timestep%d.png"%timestep)
    plt.clf()