import numpy as np
import torch

from mbmf.training.normalizer import Normalizer

class Buffer(object):
    def __init__(
        self,
        state_size,
        action_size,
        ensemble_size,
        batch_size,
        normalizer=None,
        buffer_size=10 ** 6,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros((buffer_size, 1))
        self.next_states = np.zeros((buffer_size, state_size))
        self.state_deltas = np.zeros((buffer_size, state_size))
        self.not_dones = np.zeros((buffer_size, 1))

        self.normalizer = normalizer
        self._total_steps = 0

    def add(self, state, action, reward, next_state, done):
        idx = self._total_steps % self.buffer_size
        state_delta = next_state - state

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_deltas[idx] = state_delta
        self.next_states[idx] = next_state
        self.not_dones[idx] = not done
        self._total_steps += 1

        if self.normalizer is not None:
            self.normalizer.update(state, action, state_delta)

    def train_batches(self, batch_size):
        indices = [
            np.random.permutation(range(self.current_size)) for _ in range(self.ensemble_size)
        ]
        indices = np.stack(indices).T

        for i in range(0, self.current_size, batch_size):
            j = min(self.current_size, i + batch_size)
            if (j - i) < batch_size and i != 0:
                return
            batch_size = j - i

            batch_indices = indices[i:j]
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            state_deltas = torch.from_numpy(state_deltas).float().to(self.device)

            states = states.reshape(self.ensemble_size, batch_size, self.state_size)
            actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
            rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
            state_deltas = state_deltas.reshape(self.ensemble_size, batch_size, self.state_size)

            yield states, actions, rewards, state_deltas

    
    def sample_proprio(self):
        idxs = np.random.randint(0, self.current_size, size=self.batch_size)
        states = torch.as_tensor(self.states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).float()
        return states, actions, rewards, next_states, not_dones


    @property
    def current_size(self):
        return min(self._total_steps, self.buffer_size)