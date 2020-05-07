import torch
import torch.nn as nn
from torch.distributions import Normal

from mbmf.models.layers import EnsembleLinearLayer

def swish(x):
    return x * torch.sigmoid(x)

class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        ensemble_size,
        hidden_size,
        normalizer=None,
        act_fn="swish",
        device="cpu",
    ):
        super().__init__()

        self.fc_1 = EnsembleLinearLayer(
            in_size, hidden_size, ensemble_size, init_type="xavier_uniform"
        )
        self.fc_2 = EnsembleLinearLayer(
            hidden_size, hidden_size, ensemble_size, init_type="xavier_uniform"
        )
        self.fc_3 = EnsembleLinearLayer(
            hidden_size, hidden_size, ensemble_size, init_type="xavier_uniform"
        )
        self.fc_4 = EnsembleLinearLayer(
            hidden_size, out_size * 2, ensemble_size, init_type="xavier_normal"
        )

        self.in_size = in_size
        self.out_size = out_size
        self.ensemble_size = ensemble_size
        self.normalizer = normalizer
        self.max_logvar = -1
        self.min_logvar = -5
        self.device = device
        self.to(device)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, states, actions):
        norm_states, norm_actions = self._preprocess_model_inputs(states, actions)
        norm_delta_mean, norm_delta_var = self._propagate_network(norm_states, norm_actions)
        delta_mean, delta_var = self._postprocess_model_outputs(norm_delta_mean, norm_delta_var)
        return delta_mean, delta_var

    def forward_agents(self, states, actions):
        """ Apply transitions from an ensemble of agents

        NOTE we need this loop unless `n_sac_agents == ensemble_size`

            - `states`: (n_agents, state_dim)
            - `actions`: (n_agents, action_dim)

        Returns
        -------
        - `next_states`: (n_agents, ensemble_size, batch_size, state_dim)

        """

        n_agents = actions.shape[0]
        agent_states = torch.zeros((n_agents, self.ensemble_size, 1, self.out_size))

        for a in range(n_agents):
            _states = states[a, :]
            _states = _states.unsqueeze(0).repeat(self.ensemble_size, 1, 1).to(self.device)
            _actions = actions[a, :]
            _actions = _actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1).to(self.device)
            
            # (ensemble_size, batch_size, out_size)
            delta_mean, delta_var = self(_states, _actions)
            next_states = _states + delta_mean
            agent_states[a, :, :, :] = next_states

        return agent_states


    def loss(self, states, actions, state_deltas):
        states, actions = self._preprocess_model_inputs(states, actions)
        delta_targets = self._preprocess_model_targets(state_deltas)
        delta_mu, delta_var = self._propagate_network(states, actions)
        loss = (delta_mu - delta_targets) ** 2 / delta_var + torch.log(delta_var)
        loss = loss.mean(-1).mean(-1).sum()
        return loss

    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_4.reset_parameters()
        self.to(self.device)

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        
        op = self.fc_1(inp)
        op = swish(op)
        op = self.fc_2(op)
        op = swish(op)
        op = self.fc_3(op)
        op = swish(op)
        op = self.fc_4(op)

        delta_mean, delta_logvar = torch.split(op, op.size(2) // 2, dim=2)
        delta_logvar = torch.sigmoid(delta_logvar)
        delta_logvar = (self.min_logvar + (self.max_logvar - self.min_logvar) * delta_logvar)
        delta_var = torch.exp(delta_logvar)

        return delta_mean, delta_var

    def _preprocess_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _preprocess_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)
        if self.normalizer is not None:
            state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _postprocess_model_outputs(self, delta_mean, delta_var):
        if self.normalizer is not None:
            delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
            delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)
        return delta_mean, delta_var
   

class RewardModel(nn.Module):
    def __init__(self, in_size, hidden_size, act_fn="relu", device="cpu"):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.device = device
        self.act_fn = getattr(F, act_fn)
        self.reset_parameters()
        self.to(device)

    def forward(self, states, actions):
        inp = torch.cat((states, actions), dim=-1)
        reward = self.act_fn(self.fc_1(inp))
        reward = self.act_fn(self.fc_2(reward))
        reward = self.fc_3(reward).squeeze(dim=1)
        return reward

    def loss(self, states, actions, rewards):
        r_hat = self(states, actions)
        return F.mse_loss(r_hat, rewards)

    def reset_parameters(self):
        self.fc_1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_3 = nn.Linear(self.hidden_size, 1)
        self.to(self.device)
   