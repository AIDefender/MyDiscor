from collections import deque
import numpy as np
from numpy.lib.arraysetops import isin
import torch
import cProfile
import time


class NStepBuffer:

    def __init__(self, gamma=0.99, nstep=3):
        assert isinstance(gamma, float) and 0 < gamma < 1.0
        assert isinstance(nstep, int) and nstep > 0

        self._discounts = [gamma ** i for i in range(nstep)]
        self._nstep = nstep
        self.reset()

    def append(self, state, action, reward):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)

    def get(self):
        assert len(self._rewards) > 0

        state = self._states.popleft()
        action = self._actions.popleft()
        reward = self._nstep_reward()
        return state, action, reward

    def _nstep_reward(self):
        reward = np.sum([
            r * d for r, d in zip(self._rewards, self._discounts)])
        self._rewards.popleft()
        return reward

    def reset(self):
        self._states = deque(maxlen=self._nstep)
        self._actions = deque(maxlen=self._nstep)
        self._rewards = deque(maxlen=self._nstep)

    def is_empty(self):
        return len(self._rewards) == 0

    def is_full(self):
        return len(self._rewards) == self._nstep

    def __len__(self):
        return len(self._rewards)


class ReplayBuffer:

    def __init__(self, memory_size, state_shape, action_shape, gamma=0.99,
                 nstep=1, arbi_reset=False, *args, **kwargs):
        assert isinstance(memory_size, int) and memory_size > 0
        assert isinstance(state_shape, tuple)
        assert isinstance(action_shape, tuple)
        assert isinstance(gamma, float) and 0 < gamma < 1.0
        assert isinstance(nstep, int) and nstep > 0

        self._memory_size = memory_size
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._gamma = gamma
        self._nstep = nstep
        # If we need arbitrary reset, the output of env.sim.get_state() needs saving for further reset
        self._arbi_reset = arbi_reset

        self._reset()

    def _reset(self):
        self._n = 0
        self._p = 0

        self._states = np.empty(
            (self._memory_size, ) + self._state_shape, dtype=np.float32)
        self._next_states = np.empty(
            (self._memory_size, ) + self._state_shape, dtype=np.float32)
        self._actions = np.empty(
            (self._memory_size, ) + self._action_shape, dtype=np.float32)

        self._rewards = np.empty((self._memory_size, 1), dtype=np.float32)
        self._dones = np.empty((self._memory_size, 1), dtype=np.float32)
        if self._arbi_reset:
            self._sim_states = [None] * self._memory_size

        if self._nstep != 1:
            self._nstep_buffer = NStepBuffer(self._gamma, self._nstep)

    def append(self, state, action, reward, next_state, done, step=None, episode_done=None, sim_state=None):
        if self._nstep != 1:
            self._nstep_buffer.append(state, action, reward)

            if self._nstep_buffer.is_full():
                state, action, reward = self._nstep_buffer.get()
                self._append(state, action, reward, next_state, done, episode_done=episode_done)

            if done or episode_done:
                while not self._nstep_buffer.is_empty():
                    state, action, reward = self._nstep_buffer.get()
                    self._append(state, action, reward, next_state, done, episode_done=episode_done)

        else:
            self._append(state, action, reward, next_state, done, step, sim_state)

    def _append(self, state, action, reward, next_state, done, step=None, sim_state=None, episode_done=None):
        self._states[self._p, ...] = state
        self._actions[self._p, ...] = action
        self._rewards[self._p, ...] = reward
        self._next_states[self._p, ...] = next_state
        self._dones[self._p, ...] = done
        if self._arbi_reset:
            self._sim_states[self._p] = sim_state

        self._n = min(self._n + 1, self._memory_size)
        self._p = (self._p + 1) % self._memory_size

    def sample(self, batch_size, device=torch.device('cpu')):
        assert isinstance(batch_size, int) and batch_size > 0

        idxes = self._sample_idxes(batch_size)
        return self._sample_batch(idxes, batch_size, device)

    def _sample_idxes(self, batch_size):
        return np.random.randint(low=0, high=self._n, size=batch_size)

    def _sample_batch(self, idxes, batch_size, device):
        states = torch.tensor(
            self._states[idxes], dtype=torch.float, device=device)
        actions = torch.tensor(
            self._actions[idxes], dtype=torch.float, device=device)
        rewards = torch.tensor(
            self._rewards[idxes], dtype=torch.float, device=device)
        dones = torch.tensor(
            self._dones[idxes], dtype=torch.float, device=device)
        next_states = torch.tensor(
            self._next_states[idxes], dtype=torch.float, device=device)
        batch = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_states': next_states
        }
        if self._arbi_reset:
            sim_states = [self._sim_states[i] for i in idxes]
            batch.update({'sim_states': sim_states})

        return batch

    def __len__(self):
        return self._n

class TemporalPrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, memory_size, state_shape, action_shape, gamma=0.99, nstep=1,
                 arbi_reset=False):
        super().__init__(memory_size, state_shape, action_shape, gamma, nstep, arbi_reset=arbi_reset)

    def _reset(self):
        super()._reset()
        self._steps = np.empty((self._memory_size, 1), dtype=np.int64)
        self._done_cnts = np.empty((self._memory_size, 1), dtype=np.int64)
        self._cur_done_cnt = 0

    def _append(self, state, action, reward, next_state, done, step, sim_state=None, episode_done=None):
        super()._append(state, action, reward, next_state, done, step, sim_state, episode_done=episode_done)
        # We can compute mod on negative number
        self._p = (self._p - 1) % self._memory_size 
        self._steps[self._p, ...] = step
        self._done_cnts[self._p, ...] = self._cur_done_cnt
        self._p = (self._p + 1) % self._memory_size
        if done or episode_done:
            self._cur_done_cnt += 1

    def _sample_batch(self, idxes, batch_size, device):
        batch = super()._sample_batch(idxes, batch_size, device)
        steps = torch.tensor(
            self._steps[idxes], dtype=torch.int64, device=device)
        done_cnts = torch.tensor(
            self._done_cnts[idxes], dtype=torch.int64, device=device)
        batch.update({"steps": steps})
        batch.update({"done_cnts": done_cnts})
        return batch

class BackTimeBuffer(TemporalPrioritizedReplayBuffer):

    def __init__(self, memory_size, state_shape, action_shape, gamma=0.99,
                 horizon = 1000, arbi_reset=False, **kwargs):
        super().__init__(memory_size, state_shape, action_shape, gamma, nstep=1, arbi_reset=arbi_reset)
        self._horizon = horizon
        print("Using bktmbuffer")

    def _reset(self):
        super()._reset()
        self.cur_traj_step = 0

    def append(self, state, action, reward, next_state, done, step=None, episode_done=None, sim_state=None):
        if self._p >= self.cur_traj_step:
            self._steps[self._p-self.cur_traj_step:self._p] += 1
        else:
            # one part of the traj is at the end of the buffer and the other part is at the beginning
            self._steps[self._p-self.cur_traj_step:] += 1
            self._steps[:self._p] += 1
        self._append(state, action, reward, next_state, done, 0, sim_state)        
        self.cur_traj_step += 1
        if done or episode_done or step > self._horizon:
            self.cur_traj_step = 0

def test_buffer():
    # for buffer in [TemporalPrioritizedReplayBuffer(100, (1,), (1,), horizon=12, backward=True), BackTimeBuffer(100, (1,), (1,), horizon=12)]:
    t1 = time.time()
    buffer = BackTimeBuffer(100000, (1,), (1,), horizon=1000)
    for i in range(300000):
        buffer.append(i, 1, 1, i+1, 0, i % 1000, i==1000)
    buffer.sample(128)
    print(time.time() - t1)
    t1 = time.time()
    buffer = TemporalPrioritizedReplayBuffer(10000, (1,), (1,), horizon=1000, backward=True)
    for i in range(300000):
        buffer.append(i, 1, 1, i+1, 0, i % 1000, i==1000)
    buffer.sample(128)
    print(time.time() - t1)
        # data = buffer.sample(15)
    t1 = time.time()
    buffer = ReplayBuffer(10000, (1,), (1,), horizon=1000, temperature=3e3)
    for i in range(300000):
        buffer.append(i, 1, 1, i+1, 0, i % 1000, i==1000)
    buffer.sample(128)
    print(time.time() - t1)
        # for i in data:
        #     print([int(i) for i in list(i.detach().cpu().numpy().reshape(-1))])

if __name__ == '__main__':
    test_buffer()