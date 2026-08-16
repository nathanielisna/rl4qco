#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy.py

This script contains the policy class.

__author__ = ""
__email__ = ""
"""
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


class FCNActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # for scaling input features
        # maximum value of all possible transformations (0-7)
        self.normalize_divisor = kwargs.pop("normalize_divisor", 7.0)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        '''
        Define the network layers
        '''
        if isinstance(self.observation_space, dict) or hasattr(self.observation_space, "spaces"):
            obs_space = self.observation_space["obs"]
        else:
            obs_space = self.observation_space
        c = obs_space.shape[0]

        # initial feature extraction
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(c, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # actor specific head
        self.policy_head = nn.Sequential(
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # three transformation rules = 3 channels
            nn.Conv2d(7, 3, kernel_size=1),
        )

        # critic specific head
        self.value_head = nn.Sequential(
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(7, 1, kernel_size=1),
        )
        self.value_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def _split_obs_mask(self, obs):
        '''
        Split the observation dictionary output of environment into the
        circuit observation image and action mask
        '''
        if isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                return obs, None
            if isinstance(obs[0], dict):
                obs_list = [o["obs"] for o in obs]
                mask_list = [o.get("mask", None) for o in obs]
                return self._stack_obs(obs_list), self._stack_masks(mask_list)
            return self._stack_obs(obs), None
        if isinstance(obs, dict):
            return obs["obs"], obs.get("mask", None)
        return obs, None

    def _stack_obs(self, obs_list):
        '''
        Takes the list of observation from multiple env and stacks it
        into a tensor along a new dimension.
        '''
        tensors = []
        for o in obs_list:
            # convert each observation into tensor
            if isinstance(o, th.Tensor):
                t = o
            else:
                t = th.as_tensor(o)

            # likely unused
            # TODO investigate whether is unutlized and should be removed
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)

            tensors.append(t)
        return th.stack(tensors, dim=0)

    def _stack_masks(self, mask_list):
        '''
        Takes in a list of masks and from multiple env and stacks it into
        a tensor along a new dimension. 
        '''
        tensors = []
        for m in mask_list:
            if m is None:
                t = th.zeros_like(mask_list[0])
            else:
                t = m if isinstance(m, th.Tensor) else th.as_tensor(m)
            tensors.append(t)

        return th.stack(tensors, dim=0)

    def _features(self, obs) -> th.Tensor:
        '''
        Takes the observation of the environment and extracts image
        features by passing it through the shared_cnn. Return the features.
        '''
        # extract image observation
        obs, _ = self._split_obs_mask(obs)

        # if integer format, which the observation is, normalize into float
        # b/w 0-1
        if obs.dtype == th.uint8:
            obs = obs.float().div_(self.normalize_divisor)
        else:
            obs = obs.float()

        # add batch dimension if missing (single observation)
        if obs.ndim == 3:  # (C,H,W)
            obs = obs.unsqueeze(0)  # becomes (1,C,H,W)

        return self.shared_cnn(obs)  # returns (B, channels', H, W)

    def _policy_logits_map(self, features: th.Tensor) -> th.Tensor:
        '''
        Passes the features through the policy head and returns the
        output/transformation action tensor
        '''
        return self.policy_head(features)

    def _policy_logits(self, logits_map: th.Tensor) -> th.Tensor:
        '''
        Flatten all dimensions except the batch dimension
        (batch dim, 3 (transformation rules) * qubit_num * qubit_dim)
        '''
        # if single observation without batch, add batch dim
        if logits_map.ndim == 3:  # (C,H,W)
            logits_map = logits_map.unsqueeze(0)  # (1,C,H,W)
        return logits_map.flatten(1)  # (B, 3*H*W)

    def _value(self, features: th.Tensor) -> th.Tensor:
        '''
        Passes feature through value head. Return the flattened value output.
        '''
        value_map = self.value_head(features)
        return self.value_pool(value_map).flatten(1)

    def _apply_mask(self, logits_map: th.Tensor, mask):
        '''
        Apply action mask on features.
        '''
        if mask is None:
            return logits_map

        # convert mask to tensor on same device
        if not isinstance(mask, th.Tensor):
            mask = th.as_tensor(mask, device=logits_map.device)
        if mask.dtype != th.bool:
            mask = mask.to(dtype=th.bool)

        # ensure batch dimension / shape alignment
        if mask.ndim == 1:  # flattened mask for single obs
            mask = mask.view_as(logits_map[0]).unsqueeze(0)
        elif mask.ndim == 2:  # flattened mask for batch
            if mask.shape[1] == logits_map[0].numel():
                mask = mask.view(mask.shape[0], *logits_map.shape[1:])
        elif mask.ndim == 3:  # single observation: (C,H,W)
            mask = mask.unsqueeze(0)  # (1,C,H,W)

        mask = mask.to(device=logits_map.device)

        # turn off invalid action neurons
        return logits_map.masked_fill(mask == 0, -1e9)

    def _trim_tensor(self, obs_tensor, mask, depth_array):
        max_depth = depth_array.max()
        return obs_tensor[:, :, :, :max_depth], mask[:, :, :, :max_depth]

    def forward(self, obs, deterministic: bool = False):
        '''
        Extract the observation and mask. Pass through shared_cnn/feature extractor.
        Pass features through policy and value. Flatten the policy output to logits.
        Select highest activated neuron.
        '''
        obs_tensor, mask = self._split_obs_mask(obs)
        # obs_tensor, mask = self._trim_tensor(obs_tensor, mask, obs["circ_eff_depth"])
        features = self._features(obs_tensor)

        # Logit map (nenv, transform_num_dim, qubit_dim, depth_dim)
        logits_map = self._policy_logits_map(features)
        logits_map = self._apply_mask(logits_map, mask)

        # Flatten
        logits = self._policy_logits(logits_map)
        dist = Categorical(logits=logits)  # normalize probability distribution

        # sample from dist
        if deterministic:
            actions = dist.probs.argmax(dim=1)
        else:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)

        value = self._value(features)
        return actions, value, log_prob

    def get_distribution(self, obs):
        '''
        Convert the CNN output logit into a Categorical Distribution.
        '''
        obs_tensor, mask = self._split_obs_mask(obs)
        features = self._features(obs_tensor)
        logits_map = self._policy_logits_map(features)
        logits_map = self._apply_mask(logits_map, mask)
        logits = self._policy_logits(logits_map)

        dist = CategoricalDistribution(action_dim=logits.shape[-1])
        dist = dist.proba_distribution(action_logits=logits)
        return dist

    def _predict(self, obs, deterministic: bool = False):
        '''
        Sample distribution for an action.
        '''
        dist = self.get_distribution(obs)
        return dist.probs.argmax(dim=1) if deterministic else dist.sample()

    def evaluate_actions(self, obs, actions: th.Tensor):
        '''
        Evaluate the selected action.
        '''
        obs_tensor, mask = self._split_obs_mask(obs)
        features = self._features(obs_tensor)

        logits_map = self._policy_logits_map(features)
        logits_map = self._apply_mask(logits_map, mask)
        logits = self._policy_logits(logits_map)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self._value(features)
        return value, log_prob, entropy

    def predict_values(self, obs):
        '''
        Predict the value
        '''
        obs_tensor, _ = self._split_obs_mask(obs)
        features = self._features(obs_tensor)
        return self._value(features)
