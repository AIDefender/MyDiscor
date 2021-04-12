import gym
from gym.envs.registration import register
from gym import spaces
import numpy as np
import copy

class ContinuousGridEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, size = 5., act_bound = 3., dimention = 2):
    super(ContinuousGridEnv, self).__init__()
    self.action_space = spaces.Box(low = np.array([-act_bound] * dimention), high = np.array([act_bound] * dimention))
    self.observation_space = spaces.Box(low = np.array([-size] * dimention), high = np.array([size] * dimention), dtype = np.float64)

    self.size = size 
    self.act_bound = act_bound
    self.dimention = dimention
    self._max_episode_steps = 100

  def step(self, action):
    if isinstance(action, list) or isinstance(action, tuple):
        action = np.array(action)
    assert isinstance(action, np.ndarray)
    action = np.min(np.vstack((action, [self.act_bound] * self.dimention)), axis = 0)
    action = np.max(np.vstack((action, [-self.act_bound] * self.dimention)), axis = 0)
    self.agent_pos = self.agent_pos + action
    self.agent_pos = np.min(np.vstack((self.agent_pos, [self.size] * self.dimention)), axis = 0)
    self.agent_pos = np.max(np.vstack((self.agent_pos, [-self.size] * self.dimention)), axis = 0)

    if ([self.size - 1] * self.dimention < self.agent_pos).all() and (self.agent_pos <= [self.size] * self.dimention).all():
        reward = 1
        done = 1
    else:
        reward = -1
        done = 0
    return copy.deepcopy(self.agent_pos), reward, done, {}

  def reset(self):
    self.agent_pos = np.array([-self.size] * self.dimention)
    return copy.deepcopy(self.agent_pos)

register(
    id='ContinuousGrid-v0',
    entry_point=ContinuousGridEnv,
    reward_threshold=1
)

if __name__ == '__main__':
    env = gym.make("ContinuousGrid-v0")
    print(env.reset())
    print(env.step((1,1)))
    print(env.step((10,10)))