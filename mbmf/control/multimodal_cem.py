import torch
import torch.nn as nn
import numpy as np
import math


def gaussian_logprob(actions, action_mean, action_std):
    print("in gaussian logprob:" , actions.shape, action_mean.shape, action_std.shape)
    return -torch.div((actions - action_mean)**2,action_std) - torch.log(action_std) #- logsqrt2pi

#okay, it's not working out here because of it so I'll do is in pseudocode.


class MpcAgent(nn.Module):
    def __init__(
        self,
        model,
        ensemble_size,
        action_size,
        plan_horizon=12,
        optimisation_iters=10,
        n_candidates=1000,
        top_candidates=100,
        reward_measure=None,
        expl_measure=None,
        use_mean=True,
        alpha=0.25,
        entropy_regularised=True,
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.ensemble_size = ensemble_size
        self.action_size = action_size

        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.n_candidates = n_candidates
        self.top_candidates = top_candidates
        self.alpha = alpha

        self.reward_measure = reward_measure
        self.expl_measure = expl_measure
        self.use_exploration = False if expl_measure is None else True
        self.use_reward = False if reward_measure is None else True
        self.entropy_regularised = entropy_regularised

        self.use_mean = use_mean
        self.device = device
        self.to(device)

#so I have ot get this focused and done... argh! or alec will do it. I'll try and do it and push it forwards
#perhaps in combination with him
    def agent_array_to_list(self, actions, agent_nums):
        L = len(agent_nums)
        tot_idx = 0
        agent_list = [torch.empty() for i in range(L)]
        for i in range(L):
            new_idx = tot_idx + agent_nums[i]
            agent_list[i] = actions[:,tot_idx:new_idx,:]
            tot_idx = new_idx
        return agent_list
    
    def agent_list_to_array(self,action_list,agent_nums):
        L = len(agent_nums)
        tot_idx = 0
        action_array = torch.zeros(self.plan_horizon,self.n_candidates,self.action_size).to(self.device)
        for i,alist in enumerate(action_list):
            new_idx = tot_idx + alist.shape[1]
            action_array[i,tot_idx:nex_idx,:,:]
            tot_idx = new_idx
        return action_array


    def forward(self, state, action_mean=None, action_std=None, is_torch=False, L=None):
        """ @TODO is_numpy - log is redundant"""

        #assume action_mean = [N_agents x plan_horizon x action_dim]
        #assume action_std = [N_agents x plan_horizon x action_dim]
        log=False
        if not is_torch:
            state = torch.from_numpy(state).float().to(self.device)
        state_size = state.size(0)

        if action_mean is None:
            action_mean = torch.zeros(self.plan_horizon, 1, self.action_size)
            action_mean = action_mean.to(self.device)
            # deal with NaN
            action_mean[action_mean != action_mean] = 0
        else:
            action_mean = action_mean.unsqueeze(dim=2).to(self.device).repeat(1,1,self.n_candidates//self.N_agents,1)
            #now [N_agents x plan_horizon x n_candidates//N_agents x action_dim]

        if action_std is None:
            action_std = torch.ones(self.plan_horizon, 1, self.action_size)
            action_std = action_std.to(self.device)
            # deal with NaN
            action_std[action_mean != action_mean] = 1.0
        else:
            log = True
            action_std = action_std.unsqueeze(dim=2).to(self.device).repeat(1,1,self.n_candidates//self.N_agents,1)
            #now [N_agents x plan_horizon x n_candidates//N_agents x action_dim]

        if log:
            L.log_cem_stats(action_mean, action_std)

        #setup the initial (uniform) distribution over action mixtues (here set to be the number of sac agents)
        agent_pis = torch.ones(self.N_agents)/ self.N_agents
        agent_nums = torch.ones(self.N_agents) * (self.n_candidates//self.N_agents)

        for i in range(self.optimisation_iters):
            print("in control optim")
            #loop over each agent in the n_actions and apply the correct action mean and std
            for i in range(agent_nums):
                actions[i] = action_mean + action_std * torch.randn(
                    self.plan_horizon, self.agent_nums[i], self.action_size, device=self.device
                )
            #turn action list into a big array
            actions = agent_list_to_array(actions,agent_nums)
            states, delta_vars, delta_means = self.perform_rollout(state, actions)
            returns = torch.zeros(self.n_candidates).float().to(self.device)

            if self.expl_measure is not None and self.use_exploration:
                expl_bonus = self.expl_measure(delta_means, delta_vars)
                returns += expl_bonus

            if self.reward_measure is not None and self.use_reward:
                #print("states: ", states.shape)
                _states = states.view(-1, state_size)
                #print("_states: ", _states.shape)
                _actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
                #print("_actions: ", _actions.shape)
                _actions = _actions.view(-1, self.action_size)
                #print("_actions", _actions.shape)

                rewards = self.reward_measure(_states, _actions)
                rewards = rewards.view(self.ensemble_size, self.plan_horizon, self.n_candidates)
                if self.entropy_regularised:
                    action_mean = action_mean.unsqueeze(0).squeeze(-1).repeat(self.ensemble_size,1,self.n_candidates)
                    action_std = action_std.unsqueeze(0).squeeze(-1).repeat(self.ensemble_size,1,self.n_candidates)
                    logqa = gaussian_logprob(actions.squeeze(-1),action_mean, action_std)
                    rewards -= logqa
                rewards = rewards.mean(dim=1).sum(dim=0)
                returns += rewards

            #get the topk threshold to apply for each batch
            returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
            _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=True)
            threshold = topk[-1] # get the lowest reward threshold required
            return_list = agent_array_to_list(returns, agent_nums)
            #only keep those in the list with returns above the threshold
            action_means = torch.zeros(self.N_agents, self.action_size)
            action_stds = torch.zeros(self.N_agents, self.action_size)
            tot_len = 0
            for i,rlist in enumerate(return_list):
                accept_idxs = torch.where(rlist[i] >= threshold)
                rlist[i] = rlist[i][:,accept_idxs,:]
                #fit the gaussians to the points which is necessary
                action_means[i,:] = actions[:,:,i,:].mean(dim=(0,1))
                action_stds[i,] = actions[:,:,i,:].std(dim=1).mean(dim=0)
                #set the new agent num as the number of surviving elements from that agent
                agent_nums[i] = len(accept_idxs)
                #and accumulate the total_len so we can get probabilities 
                tot_len += len(accept_idxs)
            #get total probability
            agent_pis = agent_nums / total_len
            #then regenerate agent nums to the full n candidates 
            agent_nums = int(agent_pis * self.n_candidates)
            
            new_action_mean, new_action_std = self._fit_gaussian(actions, returns)
            action_mean = self.alpha * action_mean + (1 - self.alpha) * new_action_mean
            action_std = self.alpha * action_std + (1 - self.alpha) * new_action_std
            L.log_cem_stats(action_mean, action_std)

        L.flush_cem_stats()
        
        action = action_mean[0].squeeze(dim=0)
        return action.cpu().detach().numpy()

    def perform_rollout(self, current_state, actions):
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T

        current_state = current_state.unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(self.ensemble_size, self.n_candidates, 1)
        states[0] = current_state

        actions = actions.unsqueeze(0)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.model(states[t], actions[t])
            if self.use_mean:
                states[t + 1] = states[t] + delta_mean
            else:
                states[t + 1] = states[t] + self.model.sample(delta_mean, delta_var)
            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        return states, delta_vars, delta_means

    def _fit_gaussian(self, actions, returns):
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)
        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, self.top_candidates, self.action_size
        )
        action_mean, action_std_dev = (
            best_actions.mean(dim=1, keepdim=True),
            best_actions.std(dim=1, unbiased=False, keepdim=True),
        )
        return action_mean, action_std_dev
