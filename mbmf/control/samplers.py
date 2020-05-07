import time
from copy import deepcopy

import torch
import numpy as np

from mbmf import utils

class BaseSampler(object):
    def __init__(self, env):
        self._env = env

    def select_action(self, state):
        raise NotImplementedError

    def sample_record_episodes(
        self, n_episodes, buffer, action_noise=None, log_fn=None, log_every=1
    ):
        rewards = []
        steps = []
        times = []

        for _ in range(n_episodes):
            epi_reward = 0
            epi_steps = 0
            epi_time = time.time()

            state = self._env.reset()
            done = False
            float_done = False
            while not done:
                action = self.select_action(state)
                if action_noise is not None and action_noise > 0.0:
                    action = self._add_action_noise(action, action_noise)
                next_state, reward, done, _ = self._env.step(action)

                epi_reward += reward
                epi_steps += 1

                if log_fn is not None and epi_steps % log_every == 0:
                    log_fn(epi_steps, epi_reward)

                if done:
                    float_done = True

                buffer.add(state, action, reward, next_state, float(float_done))
                state = deepcopy(next_state)

                if done:
                    break

            rewards.append(epi_reward)
            steps.append(epi_steps)
            times.append(time.time() - epi_time)

        return buffer, {"rewards": rewards, "steps": steps, "times": times}

    def sample_episodes(
        self, n_episodes, action_noise=None, log_fn=None, log_every=25
    ):
        rewards = []
        steps = []
        times = []

        for _ in range(n_episodes):
            epi_reward = 0
            epi_steps = 0
            epi_time = time.time()

            state = self._env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                if action_noise is not None and action_noise > 0.0:
                    action = self._add_action_noise(action, action_noise)
                next_state, reward, done, _ = self._env.step(action)
                state = deepcopy(next_state)

                epi_reward += reward
                epi_steps += 1

                if log_fn is not None and epi_steps % log_every == 0:
                    log_fn(epi_steps, epi_reward)

                if done:
                    break

            rewards.append(epi_reward)
            steps.append(epi_steps)
            times.append(time.time() - epi_time)

        return {"rewards": rewards, "steps": steps, "times": times}

    def _add_action_noise(self, action, noise):
        action = action + np.random.normal(0, noise, action.shape)
        return action


class RandomSampler(BaseSampler):
    def __init__(self, env):
        super().__init__(env)

    def select_action(self, state):
        return self._env.sample_action()


class ControlSampler(BaseSampler):
    def __init__(self, env, control_fn):
        super().__init__(env)
        self._control_fn = control_fn

    def select_action(self, state):
        action = self._control_fn(state)
        return action
