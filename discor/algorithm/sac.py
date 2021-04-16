import copy
import os
import numpy as np
import torch
from torch.optim import Adam

from .base import Algorithm
from discor.network import TwinnedStateActionFunction, GaussianPolicy
from discor.utils import disable_gradients, soft_update, update_params, \
    assert_action


class SAC(Algorithm):

    def __init__(self, state_dim, action_dim, device, gamma=0.99, nstep=1,
                 policy_lr=0.0003, q_lr=0.0003, entropy_lr=0.0003, 
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                 target_update_coef=0.005, log_interval=10, seed=0, 
                 env=None, eval_tper=False, log_dir=None, eval_tper_interval=5e4):
        super().__init__(
            state_dim, action_dim, device, gamma, nstep, log_interval, seed)

        # Build networks.
        self._policy_net = GaussianPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
            ).to(self._device)
        self._online_q_net = TwinnedStateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
            ).to(self._device)
        self._target_q_net = TwinnedStateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
            ).to(self._device).eval()

        # Copy parameters of the learning network to the target network.
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self._target_q_net)

        # Optimizers.
        self._policy_optim = Adam(self._policy_net.parameters(), lr=policy_lr)
        self._q_optim = Adam(self._online_q_net.parameters(), lr=q_lr)

        # Target entropy is -|A|.
        self._target_entropy = -float(self._action_dim)

        # We optimize log(alpha), instead of alpha.
        self._log_alpha = torch.zeros(
            1, device=self._device, requires_grad=True)
        self._alpha = self._log_alpha.detach().exp()
        self._alpha_optim = Adam([self._log_alpha], lr=entropy_lr)

        self._target_update_coef = target_update_coef

        self._env = env
        self._eval_tper = eval_tper
        self._eval_tper_interval = eval_tper_interval
        self._log_dir = log_dir
        self._stats_dir = os.path.join(log_dir, "stats")

    def explore(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action, _, _ = self._policy_net(state)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action

    def exploit(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            _, _, action = self._policy_net(state)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action

    def update_target_networks(self):
        soft_update(
            self._target_q_net, self._online_q_net, self._target_update_coef)

    def update_online_networks(self, batch, writer):
        self._learning_steps += 1
        batch = batch["uniform"]
        self.update_policy_and_entropy(batch, writer)
        self.update_q_functions(batch, writer)

    def update_policy_and_entropy(self, batch, writer):
        states = batch["states"]

        # Update policy.
        policy_loss, entropies = self.calc_policy_loss(states)
        update_params(self._policy_optim, policy_loss)

        # Update the entropy coefficient.
        entropy_loss = self.calc_entropy_loss(entropies)
        update_params(self._alpha_optim, entropy_loss)
        self._alpha = self._log_alpha.detach().exp()

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'loss/entropy', entropy_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/alpha', self._alpha.item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self._learning_steps)

    def calc_policy_loss(self, states):
        # Resample actions to calculate expectations of Q.
        sampled_actions, entropies, _ = self._policy_net(states)

        # Expectations of Q with clipped double Q technique.
        qs1, qs2 = self._online_q_net(states, sampled_actions)
        qs = torch.min(qs1, qs2)

        # Policy objective is maximization of (Q + alpha * entropy).
        assert qs.shape == entropies.shape
        policy_loss = torch.mean((- qs - self._alpha * entropies))

        return policy_loss, entropies.detach_()

    def calc_entropy_loss(self, entropies):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self._log_alpha * (self._target_entropy - entropies))
        return entropy_loss

    def update_q_functions(self, batch, writer, imp_ws1=None, imp_ws2=None, fast_batch=None, err_preds=None):
        states, actions, rewards, next_states, dones = \
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["dones"]

        # Calculate current and target Q values.
        curr_qs1, curr_qs2 = self.calc_current_qs(states, actions)
        target_qs = self.calc_target_qs(rewards, next_states, dones)

        # Update Q functions.
        q_loss, mean_q1, mean_q2, unweighted_q_loss = \
            self.calc_q_loss(curr_qs1, curr_qs2, target_qs, imp_ws1, imp_ws2)
        update_params(self._q_optim, q_loss)

        #TODO: compute Q loss for online batch

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/Q', unweighted_q_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q1', mean_q1, self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q2', mean_q2, self._learning_steps)

        if self._eval_tper and self._learning_steps % self._eval_tper_interval == 0:
            steps = batch["steps"]
            sim_states = batch["sim_states"]
            self.eval_Q(states[:128], actions[:128], steps[:128], sim_states[:128], curr_qs1[:128], err_preds[:128])

        # Return their values for DisCor algorithm.
        return curr_qs1.detach(), curr_qs2.detach(), target_qs

    def eval_Q(self, states, actions, steps, sim_states, curr_qs, err_preds=None):

        Qpi = self.get_real_Q(states, actions, steps, sim_states)
        assert curr_qs.shape == Qpi.shape
        Qpi_loss = (curr_qs - Qpi) ** 2
        np.savetxt(os.path.join(self._stats_dir, "Qpi_loss_timestep%d.txt"%self._learning_steps), Qpi_loss.detach().cpu().numpy())
        np.savetxt(os.path.join(self._stats_dir, "step_timestep%d.txt"%self._learning_steps), steps.detach().cpu().numpy())
        np.savetxt(os.path.join(self._stats_dir, "Qvalue_timestep%d.txt"%self._learning_steps), curr_qs.detach().cpu().numpy())
        if err_preds is not None:
            np.savetxt(os.path.join(self._stats_dir, "Error_pred_timestep%d.txt"%self._learning_steps), err_preds.detach().cpu().numpy())

    
    def get_real_Q(self, states, actions, steps, sim_states, eval_cnt = 10):
        batch_size = states.shape[0]
        envs = [copy.deepcopy(self._env) for _ in range(batch_size)]
        print("Evaluating real Q loss on %d samples"%batch_size)
        all_Qpi = []
        for i in range(eval_cnt):
            print("Evaluating: count %d"%i)
            origin_obs = [env.reset() for env in envs]
            [env.sim.set_state(s) for (env, s) in zip(envs, sim_states)]
            dones = [False] * batch_size
            cur_states = copy.deepcopy(states)
            this_Qpi = None
            this_gamma = 1
            for i in range(self._env._max_episode_steps - int(torch.min(steps))):
                if i == 0:
                    next_actions = actions
                else:
                    next_actions, *_ = self._policy_net(cur_states)
                next_actions = next_actions.detach().cpu().numpy()
                # res = [env.step(a) for (env, a) in zip(envs, next_actions)]
                next_obs = []
                rewards = []
                new_done = []
                for index, (env, a) in enumerate(zip(envs, next_actions)):
                    if dones[index]:
                        rewards.append(0)
                        next_obs.append(origin_obs[index]) # any obs is ok
                        new_done.append(True)
                    else:
                        ns, reward, done, _ = env.step(a)
                        rewards.append(reward)
                        next_obs.append(ns)
                        new_done.append(done)

                if this_Qpi is not None:
                    this_Qpi = this_Qpi + this_gamma * np.array(rewards)
                else:
                    this_Qpi = np.array(rewards)
                cur_states = torch.tensor(next_obs, dtype=torch.float32).to(device=self._device)
                dones = copy.deepcopy(new_done)
                this_gamma *= self._gamma
                if sum(dones) == states.shape[0]:
                    break
            all_Qpi.append(this_Qpi)
        Q_pi = torch.tensor(np.mean(all_Qpi, axis=0)).to(device=self._device).reshape(-1,1)

        return Q_pi

    def calc_current_qs(self, states, actions):
        curr_qs1, curr_qs2 = self._online_q_net(states, actions)
        return curr_qs1, curr_qs2

    def calc_target_qs(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self._policy_net(next_states)
            next_qs1, next_qs2 = self._target_q_net(next_states, next_actions)
            next_qs = \
                torch.min(next_qs1, next_qs2) + self._alpha * next_entropies

        assert rewards.shape == next_qs.shape
        target_qs = rewards + (1.0 - dones) * self._discount * next_qs

        return target_qs

    def calc_q_loss(self, curr_qs1, curr_qs2, target_qs, imp_ws1=None,
                    imp_ws2=None):
        assert imp_ws1 is None or imp_ws1.shape == curr_qs1.shape
        assert imp_ws2 is None or imp_ws2.shape == curr_qs2.shape
        assert not target_qs.requires_grad
        assert curr_qs1.shape == target_qs.shape

        # Q loss is mean squared TD errors with importance weights.
        if imp_ws1 is None:
            q1_loss = torch.mean((curr_qs1 - target_qs).pow(2))
            q2_loss = torch.mean((curr_qs2 - target_qs).pow(2))
        else:
            q1_loss = torch.mean((curr_qs1 - target_qs).pow(2) * imp_ws1)
            q2_loss = torch.mean((curr_qs2 - target_qs).pow(2) * imp_ws2)

        # Mean Q values for logging.
        mean_q1 = curr_qs1.detach().mean().item()
        mean_q2 = curr_qs2.detach().mean().item()

        # for a fair comparison
        unweighted_q_loss = torch.mean((curr_qs1 - target_qs).pow(2)) + torch.mean((curr_qs2 - target_qs).pow(2))

        return q1_loss + q2_loss, mean_q1, mean_q2, unweighted_q_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))
