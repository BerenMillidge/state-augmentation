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
        n_augments = 3,
        augment_std=0.1,
        reward_std = 0.05,
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

        self.n_augments = n_augments
        self.augment_std = augment_std
        self.reward_std=reward_std

        self.normalizer = normalizer
        self._total_steps = 0

    def base_add(self, state, action, reward, next_state, done):
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

    def add(self, state,action,reward,next_state,done):
        #add the un-jittered batch
        self.base_add(state, action,reward,next_state,done)
        #print("in add augments: ", self.n_augments)
        if self.n_augments > 0:
            #loop over augments
            for n in range(self.n_augments):
                aug_state = state + np.random.normal(0.0,self.augment_std,size=state.shape)
                aug_next_state = next_state + np.random.normal(0.0,self.augment_std,size=state.shape)
                aug_reward = reward
                if self.reward_std > 0:
                    aug_reward = reward + np.random.normal(0.0,self.reward_std, size=reward.shape)
                self.base_add(aug_state, action,aug_reward,aug_next_state,done)


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
        #all are [BATCH_SIZE x FEATURE_DIM]
        idxs = np.random.randint(0, self.current_size, size=self.batch_size)
        states = torch.as_tensor(self.states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).float()
        #print("inside sample proprio:")
        #print("idxs: ", idxs.shape)
        #print("states: ", states.shape)
        #print("actions:", actions.shape)
        #print("rewards: ", rewards.shape)
        #print("next_states: ", next_states.shape)
        #print("not_dones: ", not_dones.shape)
        return states, actions, rewards, next_states, not_dones

    #so one way to do this would be to increase the effective batch size
    #with the augmentations. The other way would be to throw out the other data?
    #or alternatively, additionally, whenever data is ADDED to the buffer
    #add in additional jittered info. Not sure what way to do it. Let's try this first method
    #here first to get reasonably good at it and see where it breaks stuff
    #which it likely doesn't right? yeah it will.

    def sample_gauss_jittered_proprio(self,n_augments, augment_std,reward_std=0):
        states, actions, rewards, next_states,not_dones = self.sample_proprio()
        states = states.repeat(n_augments,1)
        actions = actions.repeat(n_augments,1)
        rewards = rewards.repeat(n_augments,1)
        next_states = next_states.repeat(n_augments,1)
        not_dones = not_dones.repeat(n_augments,1)
        state_noise = torch.empty(states.shape).normal_(mean=0,std=augment_std)
        state_noise[0:self.batch_size,:] = torch.zeros([self.batch_size,self.state_size])
        states += state_noise #be aware this jitters the original as well
        
        next_state_noise = torch.empty(next_states.shape).normal_(mean=0,std=augment_std)
        next_state_noise[0:self.batch_size,:] = torch.zeros([self.batch_size,self.state_size])
        next_states += next_state_noise #be aware this jitters the original as well
        if reward_std > 0.0:
            reward_noise = torch.empty(rewards.shape).normal_(mean=0,std=reward_std)
            reward_noise[0:self.batch_size,:] = torch.zeros([self.batch_size,1])
            rewards += reward_noise #be aware this jitters the original as well

        return states, actions, rewards, next_states, not_dones
    @property
    def current_size(self):
        return min(self._total_steps, self.buffer_size)