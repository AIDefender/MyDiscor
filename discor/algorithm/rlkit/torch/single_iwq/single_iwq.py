from collections import OrderedDict

import numpy as np
import math
import time

import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

def weighted_mse_loss(input, target, weight):
    return torch.mean(torch.sum(weight * (input - target) ** 2, dim=1))

class SingleIWQTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf,
            target_qf,
            num_samples,
            weighted_mse,
            temperature,
            start_weighted,
            beta,
            increasing_beta,
            decreasing_beta,
            num_epochs,
            softmax,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf
        self.temperature = temperature
        self.num_samples = num_samples
        self.weighted_mse = weighted_mse 
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.start_weighted = start_weighted
        self.num_epochs = num_epochs
        self.softmax = softmax
        self.decreasing_beta = decreasing_beta
        self.increasing_beta = increasing_beta

        self.delta_beta = 1 / self.num_epochs

        if self.increasing_beta:
            self.beta = float(0)
        elif self.decreasing_beta:
            self.beta = float(1)
        else:
            self.delta_beta = 0
            self.beta = beta

        self.do_weighted_mse = self.weighted_mse and self.start_weighted == 0
        if self.do_weighted_mse:
            self.qf_criterion = nn.MSELoss(reduction='none')
        else:
            self.qf_criterion = nn.MSELoss()

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths


        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True


        # FOR DEBUGGING
        self.num_steps = 0

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        batch_size = obs.size(0)

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_all = self.qf(obs, new_obs_actions)
        q_new_actions, _ = torch.min(q_all, dim=1)

        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q_pred = self.qf(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )

        target_all = self.target_qf(next_obs, new_next_actions)
        target_q_values, _ = torch.min(target_all, dim=1)
        target_q_values = target_q_values.view(batch_size, 1)

        target_q_values = target_q_values - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.repeat(1, self.num_samples)

        if self.do_weighted_mse:
            # creating the importance weights
            qf_loss_raw = self.qf_criterion(q_pred, q_target.detach())
            mse_detach = qf_loss_raw.detach() 
            if self.softmax:
#            mse_norm = torch.mean((-mse_detach).exp(), dim=1).view(-1, 1) + 1e-6
#            weights = ((-mse_detach).exp() + 1e-6) / mse_norm
                weights = F.softmax(-mse_detach, dim=1)
            else:
                mse_norm = torch.mean((-mse_detach).exp(), dim=1).view(-1, 1) + 1e-8
                weights = ((-mse_detach).exp() + 1e-8) / mse_norm
                
#            weights = -self.qf_criterion(q_pred_detach, q_target_detach)
#            weights = F.softmax(difference / self.temperature, dim=1)
            qf_loss = self.beta * torch.mean(qf_loss_raw * weights.detach()) \
                    + (1 - self.beta) * torch.mean(qf_loss_raw)
        else:
            qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.num_steps += 1

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf, self.target_qf, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        if self.weighted_mse and epoch >= self.start_weighted:
            self.do_weighted_mse = True
            self.qf_criterion = nn.MSELoss(reduction='none')


        if self.increasing_beta:
            self.beta += self.delta_beta
        elif self.decreasing_beta:
            self.beta -= self.delta_beta

        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_qf,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf,
            target_qf1=self.qf,
        )

