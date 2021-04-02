from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.networks import FlattenMlp


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            prob_classifier: FlattenMlp,
            imp,
            q_net,
            policy_net,
            residual=False,

            discount=0.99,
            reward_scale=1.0,
            temperature=1.0,
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
        self.imp = imp
        self.residual = residual
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.prob_classifier = prob_classifier
        self.temperature = temperature
        self.q_net = q_net
        self.policy_net = policy_net

        if self.imp:
            assert self.policy_net or self.q_net, 'at least one of q or policy net need IW'

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period


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

        self.qf_criterion = nn.MSELoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.prob_classifier_optimizer = optimizer_class(
            self.prob_classifier.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
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
        Prob Clasifier
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

            if self._n_train_steps_total % 1000 == 0:
                print(loss)
    
            self.prob_classifier_optimizer.zero_grad()
            loss.backward()
            self.prob_classifier_optimizer.step()

            importance_weights = F.sigmoid(slow_preds/self.temperature).detach()
            importance_weights = importance_weights / torch.sum(importance_weights)

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

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )

        if self.imp and self.policy_net:
            policy_loss = (alpha*log_pi - q_new_actions)
            policy_loss = (policy_loss * importance_weights.detach()).sum()
        else:
            policy_loss = (alpha*log_pi - q_new_actions).mean()


        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        if self.imp and self.q_net and self.residual:
            q1_imp = (q1_pred - q_target.detach()) * importance_weights.detach() 
            q2_imp = (q2_pred - q_target.detach()) * importance_weights.detach() 
#            qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
#            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

            qf1_loss = (q1_imp ** 2).sum()
            qf2_loss = (q2_imp ** 2).sum()
#            qf1_loss = (qf1_loss * importance_weights.detach()).sum()
#            qf2_loss = (qf2_loss * importance_weights.detach()).sum()
        elif self.imp and self.q_net and not self.residual:
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

            qf1_loss = (qf1_loss * importance_weights.detach()).sum()
            qf2_loss = (qf2_loss * importance_weights.detach()).sum()
        else:
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach()).mean()
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach()).mean()

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
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

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
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
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.prob_classifier,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )

