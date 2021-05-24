import numpy as np
import matplotlib.pyplot as plt 
import copy
import seaborn as sns
sns.set_context('paper', font_scale=1.8)


STATE_CNT = 5
ACT_CNT = 3 
R = {
    0: -5, # act 0, left, reward -5
    1: 2, # act 1, stay still, reward 2
    2: 1, # act 2, right, reward 1
}
done = [{}, {}, {}]
# done for all states when act=0: all true
done[0].update([(i, 1) for i in range(STATE_CNT)]) 
# done for all states when act=1: all true
done[1].update([(i, 1) for i in range(STATE_CNT)]) 
# done for all states when act=2: all false except last state
done[2].update([(i, 0) if i != STATE_CNT - 1 else (i, 1) for i in range(STATE_CNT)])

Q_init = np.zeros((STATE_CNT - 1, ACT_CNT))
Q_star = np.array([
    [-5, 2, 4],
    [-5, 2, 3],
    [-5, 2, 2],
    [-5, 2, 1]
])

basic_lr = 0.01

def Q_iter(Qk, weight = np.ones((STATE_CNT - 1, ACT_CNT))):
    assert Qk.shape == weight.shape
    Q = copy.deepcopy(Qk)
    td = np.empty_like(Qk)
    for s in range(STATE_CNT - 1):
        for a in range(ACT_CNT):
            if done[a][s + 1]:
                Vnext = 0
            else:
                Vnext = np.max(Q[s+1])
            td[s, a] = R[a] + Vnext - Q[s, a]
            Q[s, a] = Q[s, a] + weight[s, a] * basic_lr * td[s, a]
    return Q, td

def eval(Qk, rews):
    # evaluate
    s = 0
    rew = 0
    while True:
        a = np.argmax(Qk[s])
        r = R[a]
        rew += r 
        if done[a][s+1]:
            break
        s = s + 1
    rews.append(rew)

def per():
    Qk = copy.deepcopy(Q_init)
    weight = np.ones_like(Qk)
    rews = []
    tds = []
    for i in range(300):
        Qk, td = Q_iter(Qk, weight)
        weight = 1 / (5 + np.argsort(-np.square(td), axis=None).reshape(STATE_CNT - 1, ACT_CNT))
        weight = weight / np.sum(weight) * (STATE_CNT - 1) * ACT_CNT
        if i % 30 == 0:
            tds.append(round(np.mean(np.square(td)), 2))
            eval(Qk, rews)
    print("per    :", rews)
    print("per(td)    :", tds)

    return rews, tds

def discor():
    Qk = copy.deepcopy(Q_init)
    weight = np.ones_like(Qk)
    rews = []
    tds = []
    QkQstars = []
    for i in range(300):
        Qk, td = Q_iter(Qk, weight)
        Q_subopt = np.empty_like(Qk)
        for s in range(STATE_CNT - 1):
            for a in range(ACT_CNT):
                if done[a][s+1]:
                    Q_subopt[s][a] = 0
                else:
                    Q_subopt[s][a] = Qk[s+1][a] - Q_star[s+1][a]
        weight = np.exp(-np.abs(Q_subopt * 0.5))
        weight = weight / np.sum(weight) * (STATE_CNT - 1) * ACT_CNT
        if i % 30 == 0:
            QkQstars.append(round(np.mean(np.abs(Qk-Q_star)), 2))
            tds.append(round(np.mean(np.square(td)), 2))
            eval(Qk, rews)
    print("discor    :", rews)
    print("discor(td)    :", tds)
    print("discor(Qsub) :",QkQstars)

    return rews, QkQstars

def uniform():
    Qk = copy.deepcopy(Q_init)
    rews = []
    tds = []
    QkQstars = []
    for i in range(300):
        Qk, td = Q_iter(Qk)
        if i % 30 == 0:
            QkQstars.append(round(np.mean(np.abs(Qk-Q_star)), 2))
            tds.append(round(np.mean(np.square(td)), 2))
            eval(Qk, rews)
    print("uniform:", rews)
    print("uniform(td):", tds)
    print("uniform(Qsub):",QkQstars)
    return rews, tds, QkQstars

per_rew, per_td = per()
discor_rew, discor_qkqs = discor()
uni_rew, uni_td, uni_qkqs = uniform()            
x = np.arange(0, 300, 30)

fig, ax1 = plt.subplots()
ax1.plot(x, uni_rew, color='b', label="Uniform")
ax1.plot(x, per_rew, color='r', label="PER")
ax1.set_ylabel("Reward(solid)")
ax1.legend()

ax2 = ax1.twinx() 
ax2.set_ylabel("Difference in TD error(dashed)")
ax2.plot(x, np.array(per_td) - np.array(uni_td), color='g', ls='dashed', label="PER - Uniform")
ax2.legend()
# ax2.plot(x, per_td, color='r', ls='dashed', label="per")
fig.suptitle("Suboptimality of PER")
fig.savefig("didactic_per.png", bbox_inches = 'tight', pad_inches = 0.1)

plt.cla()

fig, ax1 = plt.subplots()
ax1.plot(x, uni_rew, color='b', label="Uniform")
ax1.plot(x, discor_rew, color='r', label="DisCor")
ax1.set_ylabel("Reward(solid)")
ax1.legend()

ax2 = ax1.twinx() 
ax2.set_ylabel("Difference in |Qk-Q*|(dashed)")
# ax2.plot(x, uni_qkqs, color='b', ls='dashed', label="Uniform")
ax2.plot(x, np.array(discor_qkqs) - np.array(uni_qkqs), color='g', ls='dashed', label="DisCor - Uniform")
ax2.legend(loc='center left')
fig.suptitle("Suboptimality of discor")
fig.savefig("didactic_discor.png", bbox_inches = 'tight', pad_inches = 0.1)