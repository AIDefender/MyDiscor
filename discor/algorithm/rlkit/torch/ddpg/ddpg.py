from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DDPGTrainer(TorchTrainer):
    """
    Deep Deterministic Policy Gradient
    """
    def __init__(
            self,
            qf,
            target_qf,
            policy,
            target_policy,
            prob_classifier,
            q_net,
            imp,
            residual,
            temperature,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            min_q_value=-np.inf,
            max_q_value=np.inf,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.imp = imp
        self.residual = residual
        self.temperature = temperature
        self.q_net = q_net
        self.prob_classifier = prob_classifier

        self.qf = qf
        self.target_qf = target_qf
        self.policy = policy
        self.target_policy = target_policy

        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        assert self.qf_weight_decay == 0
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.qf_criterion = nn.MSELoss(reduction='none')
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )
        self.prob_classifier_optimizer = optimizer_class(
            self.prob_classifier.parameters(),
            lr=self.policy_learning_rate,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch, fast_batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Prob Classifier
        """
        if self.imp:
            fast_obs = fast_batch['observations']
            fast_actions = fast_batch['actions']

            slow_samples = torch.cat((obs, actions), dim=1)
            fast_samples = torch.cat((fast_obs, fast_actions), dim=1)

            zeros = torch.zeros(slow_samples.size(0))
            ones = torch.ones(fast_samples.size(0))
    
            if obs.is_cuda:
                zeros = zeros.cuda()
                ones = ones.cuda()

            slow_preds = self.prob_classifier(slow_samples)
            fast_preds = self.prob_classifier(fast_samples)

            loss = F.binary_cross_entropy(F.sigmoid(slow_preds), zeros) + \
                    F.binary_cross_entropy(F.sigmoid(fast_preds), ones)

            self.prob_classifier_optimizer.zero_grad()
            loss.backward()
            self.prob_classifier_optimizer.step()

            importance_weights = F.sigmoid(slow_preds/self.temperature).detach()
            importance_weights = importance_weights / torch.sum(importance_weights)
        

        """
        Policy operations.
        """
        if self.policy_pre_activation_weight > 0:
            policy_actions, pre_tanh_value = self.policy(
                obs, return_preactivations=True,
            )
            pre_activation_policy_loss = (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            q_output = self.qf(obs, policy_actions)
            raw_policy_loss = - q_output.mean()
            policy_loss = (
                    raw_policy_loss +
                    pre_activation_policy_loss * self.policy_pre_activation_weight
            )
        else:
            policy_actions = self.policy(obs)
            q_output = self.qf(obs, policy_actions)
            raw_policy_loss = policy_loss = - q_output.mean()

        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
        )
        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)
        q_pred = self.qf(obs, actions)
#        bellman_errors = (q_pred - q_target) ** 2
        if self.imp:
            qf_loss = self.qf_criterion(q_pred, q_target.detach())
            qf_loss = (qf_loss * importance_weights.detach()).sum()
        else:
            qf_loss = self.qf_criterion(q_pred, q_target).mean()


        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self._update_target_networks()

        """
        Save some statistics for eval using just one batch.
        """
#        if self._need_to_update_eval_statistics:
#            self._need_to_update_eval_statistics = False
#            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
#            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
#                policy_loss
#            ))
#            self.eval_statistics['Raw Policy Loss'] = np.mean(ptu.get_numpy(
#                raw_policy_loss
#            ))
#            self.eval_statistics['Preactivation Policy Loss'] = (
#                    self.eval_statistics['Policy Loss'] -
#                    self.eval_statistics['Raw Policy Loss']
#            )
#            self.eval_statistics.update(create_stats_ordered_dict(
#                'Q Predictions',
#                ptu.get_numpy(q_pred),
#            ))
#            self.eval_statistics.update(create_stats_ordered_dict(
#                'Q Targets',
#                ptu.get_numpy(q_target),
#            ))
#            self.eval_statistics.update(create_stats_ordered_dict(
#                'Bellman Errors',
#                ptu.get_numpy(bellman_errors),
#            ))
#            self.eval_statistics.update(create_stats_ordered_dict(
#                'Policy Action',
#                ptu.get_numpy(policy_actions),
#            ))
        self._n_train_steps_total += 1

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)
        else:
            if self._n_train_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_policy,
            self.target_qf,
            self.prob_classifier,
        ]

    def get_epoch_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
            trained_policy=self.policy,
            target_policy=self.target_policy,
        )
