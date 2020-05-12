import numpy as np
import torch
import torch.nn as nn

from mbmf import utils

class HybridAgent(nn.Module):
    """ Take an ensemble of SAC agents and an MPC planner """

    def __init__(self, 
                sac_agents, 
                planner, 
                ensemble_model, 
                buffer, 
                action_dim, 
                L=None, 
                stochastic=False, 
                update_sac=True, 
                warm_up=False, 
                n_sac_updates=1, 
                cem_std=1.0, 
                device='cpu'):
        super().__init__()
        self.sac_agents = sac_agents
        self.planner = planner
        self.ensemble_model = ensemble_model
        self.buffer = buffer
        self.action_dim = action_dim
        self.n_sac_updates = n_sac_updates
        self.cem_std = cem_std
        self.L = L
        self.stochastic = stochastic
        self.update_sac = update_sac
        self.warm_up = warm_up
        self.device = device
        self._global_step = 0
       
    def toggle_updates(self, update_sac):
        self.update_sac = update_sac 

    def toggle_stochastic(self, stochastic):
        self.stochastic = stochastic 

    def toggle_warm_up(self, warm_up):
        """ warm up means just init zero mean Gausian """
        self.warm_up = warm_up 


    def forward(self, state, use_stds=True):
        # TODO make use_stds a param

        # TODO
        weighting = False

        if self.warm_up:
            action = self.planner(state.squeeze(), action_mean=None, action_std=None, is_torch=torch.is_tensor(state)) 
        else:
            n_agents = len(self.sac_agents)

            if weighting:
                weights = np.zeros(n_agents)

            mus = np.zeros((self.planner.plan_horizon, n_agents, self.action_dim))
            pis = np.zeros((self.planner.plan_horizon, n_agents, self.action_dim))

            # (state_dim, ) -> (n_agents, ensemble_size, state_dim)
            state_tensor = torch.tensor(state).float().to(self.device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0).repeat(n_agents, self.planner.ensemble_size, 1)
            
            # predict actions and states
            for t in range(self.planner.plan_horizon):   

                # average across ensemble -> (n_agents, state_dim)
                avg_state_tensor = state_tensor.mean(dim=1)
 
                # FIXME convert all numpy
                np_avg_state_tensor = avg_state_tensor.detach().cpu().numpy()  
                
                # store actions taken (n_agents, action_dim)
                actions = np.zeros((n_agents, self.action_dim))

                # loop over agents
                for a, agent in enumerate(self.sac_agents):
                    # agent states -> (state_dim, )
                    agent_states = np_avg_state_tensor[a, :]

                    # select action
                    mu, pi = agent.select_and_sample_action(agent_states)

                    # store metrics
                    actions[a, :] = mu
                    mus[t, a, :] = mu
                    pis[t, a, :] = pi

                # propagate actions
                # average states -> (n_agents, state_dim)
                # actions -> (n_agents, action_dim)
                actions = torch.tensor(actions).float().to(self.device)

                if weighting:
                    # average over ensemble 
                    rewards = self.planner.reward_measure(state_tensor.mean(1).to(self.device), actions=actions)
                    weights += rewards.detach().cpu().numpy()

                # state tensor -> (n_agents, ensemble_size, batch_size, state_dim)
                state_tensor = self.ensemble_model.forward_agents(avg_state_tensor, actions).squeeze(2)
                # state tensor -> (n_agents, ensemble_size, state_dim)
                state_tensor = state_tensor.squeeze(2)
            
            # mus -> (plan_horizon, n_agents, action_dim)

            # weights are (n_agents,)
            # TODO choose single agent or use average or could weight each trajectory by cost (?)
            # mus -> (plan_horizon, n_agents, action_dim) -> (plan_horizon, action_dim) 
            if not weighting:
                action_mean = np.mean(mus, axis=1)
            else:

                action_mean = np.average(mus, axis=1, weights=weights)
            action_mean = torch.tensor(action_mean).float().to(self.device)

            # TODO
            if use_stds:
                action_std = np.std(mus, axis=1)
                action_std = torch.tensor(action_std).float().to(self.device)
                action_std = torch.clamp(action_std, -10**6, 1.0) 
            else:
                action_std = torch.ones_like(action_mean) * self.cem_std

            state = torch.tensor(state).float().squeeze().to(self.device)
            print("going into contorl")
            print("action_mean: ", action_mean.shape)
            print("action_std: ", action_std.shape)
            bib
            action = self.planner(state, action_mean=action_mean, action_std=action_std, is_torch=True, L=self.L)

        # NOTE should seperate this out 
        if self.update_sac and self.buffer is not None:
            for _ in range(self.n_sac_updates):
                for sac_agent in self.sac_agents:
                    sac_agent.update(self.buffer, None, self._global_step)

        self._global_step += 1

        return action
        

        
        