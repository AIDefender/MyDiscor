import os
import torch
from torch.optim import Adam

from .base import Algorithm
from discor.network import TwinnedDQNNet
from discor.utils import disable_gradients, soft_update, update_params, \
    assert_action


class DQN(Algorithm):

    def __init__(self, state_dim, action_dim, device, gamma=0.99, nstep=1,
                 policy_lr=0.0003, q_lr=0.0003, entropy_lr=0.0003,
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                 target_update_coef=0.005, log_interval=10, seed=0):
        super().__init__(
            state_dim, action_dim, device, gamma, nstep, log_interval, seed)

        # Build networks.
        self._online_q_net = TwinnedDQNNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
            ).to(self._device)
        self._target_q_net = TwinnedDQNNet(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
            ).to(self._device).eval()

        # Copy parameters of the learning network to the target network.
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self._target_q_net)

        # Optimizers.
        self._q_optim = Adam(self._online_q_net.parameters(), lr=q_lr)

        self._target_update_coef = target_update_coef
    
    def get_action(self, state, eps=0):
        q1, q2 = self._online_q_net(state)
        q = torch.min(q1, q2)
        a = torch.argmax(q)

        return a

    def explore(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action = self.get_action(state)
        action = action.cpu().numpy()
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
        self.update_q_functions(batch, writer)

    def update_q_functions(self, batch, writer, imp_ws1=None, imp_ws2=None, fast_batch=None):
        states, actions, rewards, next_states, dones, *_ = batch

        # Calculate current and target Q values.
        curr_qs1, curr_qs2 = self.calc_current_qs(states)
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

        # Return there values for DisCor algorithm.
        return curr_qs1.detach(), curr_qs2.detach(), target_qs

    def calc_current_qs(self, states):
        curr_qs1, curr_qs2 = self._online_q_net(states)
        return curr_qs1, curr_qs2

    def calc_target_qs(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.get_action(next_states)
            next_qs1, next_qs2 = self._target_q_net(next_states)
            next_qs = torch.min(next_qs1, next_qs2)

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
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))
