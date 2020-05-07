import torch
import torch.nn as nn


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

        self.use_mean = use_mean
        self.device = device
        self.to(device)

    def forward(self, state, action_mean=None, action_std=None, is_torch=False, L=None):
        """ @TODO is_numpy - log is redundant"""

        if not is_torch:
            state = torch.from_numpy(state).float().to(self.device)
        state_size = state.size(0)

        if action_mean is None:
            action_mean = torch.zeros(self.plan_horizon, 1, self.action_size)
            action_mean = action_mean.to(self.device)
            # deal with NaN
            action_mean[action_mean != action_mean] = 0
        else:
            action_mean = action_mean.unsqueeze(dim=1).to(self.device)

        if action_std is None:
            action_std = torch.ones(self.plan_horizon, 1, self.action_size)
            action_std = action_std.to(self.device)
            # deal with NaN
            action_std[action_mean != action_mean] = 1.0
        else:
            log = True
            action_std = action_std.unsqueeze(dim=1).to(self.device)

        if log:
            L.log_cem_stats(action_mean, action_std)

        for i in range(self.optimisation_iters):
            actions = action_mean + action_std * torch.randn(
                self.plan_horizon, self.n_candidates, self.action_size, device=self.device
            )
            states, delta_vars, delta_means = self.perform_rollout(state, actions)
            returns = torch.zeros(self.n_candidates).float().to(self.device)

            if self.expl_measure is not None and self.use_exploration:
                expl_bonus = self.expl_measure(delta_means, delta_vars)
                returns += expl_bonus

            if self.reward_measure is not None and self.use_reward:
                _states = states.view(-1, state_size)
                _actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
                _actions = _actions.view(-1, self.action_size)

                rewards = self.reward_measure(_states, _actions)
                rewards = rewards.view(self.plan_horizon, self.ensemble_size, self.n_candidates)
                rewards = rewards.mean(dim=1).sum(dim=0)
                returns += rewards

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
