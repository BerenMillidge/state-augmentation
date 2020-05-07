import os
import sys
import time
import pathlib
import argparse

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from mbmf.envs import Env
from mbmf.training import Buffer
from mbmf.control import SacAgent, EnsembleSacAgent, RandomSampler, ControlSampler
from mbmf import utils


def make_sac_agent(state_shape, action_shape, args, device):
    return SacAgent(
        obs_shape=state_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=args.sac_hidden_dim,
        discount=args.sac_discount,
        init_temperature=args.sac_init_temperature,
        alpha_lr=args.sac_alpha_lr,
        alpha_beta=args.sac_alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
    )


def main(config):
    utils.seed(config.seed)

    env = Env(
        config.env_name,
        max_episode_len=config.max_episode_len,
        action_repeat=config.action_repeat,
        seed=config.seed,
    )
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    buffer = Buffer(
        state_size,
        action_size,
        None,
        config.batch_size,
        buffer_size=config.buffer_size,
        n_augments=config.n_augments,
        augment_std=config.augment_std,
        reward_std=config.reward_std,
        device=DEVICE,
    )

    sac_agents = [
        make_sac_agent(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            args=config,
            device=DEVICE,
        )
        for _ in range(config.n_sac_agents)
    ]
    sac_agent = EnsembleSacAgent(sac_agents, buffer)

    random_sampler = RandomSampler(env)
    sac_sampler = ControlSampler(env, sac_agent)

    sac_log_fn = lambda step, reward: print(f"Collect Step {step}: {reward}")
    train_log_fn = lambda epoch, loss: print(f"Train Epoch {epoch}: {loss}")

    random_sampler.sample_record_episodes(config.n_seed_episodes, buffer)
    print(f"Collected {buffer.current_size} seed frames")

    rewards = []
    global_step = 0
    for episode in range(config.n_episodes):
        print(f"\nEpisode {episode} [{buffer.current_size} frames]")

        print(f"Collecting {config.n_collect_episodes} episodes of data")
        sac_agent.toggle_updates(True)
        sac_agent.toggle_stochastic(True)
        buffer, stats = sac_sampler.sample_record_episodes(
            config.n_collect_episodes,
            buffer,
            log_fn=sac_log_fn,
            log_every=config.sac_log_every,
        )
        print(f"Train reward: {stats['rewards']} Steps: {stats['steps']}")

        print(f"Testing on {config.n_test_episodes} episodes")
        sac_agent.toggle_updates(False)
        sac_agent.toggle_stochastic(False)
        stats = sac_sampler.sample_episodes(
            config.n_eval_episodes,
            log_fn=sac_log_fn,
            log_every=config.sac_log_every
        )
        print(f"Test reward: {stats['rewards']} steps: {stats['steps']}")
        rewards.append(stats["rewards"][0])

    return rewards


if __name__ == "__main__":

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(DEVICE)

    parser = argparse.ArgumentParser()

    # Experimental
    parser.add_argument("--n_sac_agents", type=int, default=1)
    parser.add_argument("--train_model", type=int, default=0)

    # Environment
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--env_name", type=str, default="HalfCheetahRun")
    parser.add_argument("--max_episode_len", type=int, default=100)
    parser.add_argument("--action_repeat", type=int, default=2)

    # Experiment
    parser.add_argument("--n_episodes", type=int, default=10**6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warm_start", type=int, default=0)
    parser.add_argument("--n_seed_episodes", type=int, default=10)
    parser.add_argument("--n_collect_episodes", type=int, default=1)
    parser.add_argument("--n_test_episodes", type=int, default=1)
    parser.add_argument("--n_eval_episodes", type=int, default=1)

    # Model
    parser.add_argument("--hidden_size", type=int, default=400)
    parser.add_argument("--ensemble_size", type=int, default=10)

    # Training
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--n_train_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--grad_clip_norm", type=float, default=1000)

    # Planning
    parser.add_argument("--plan_horizon", type=int, default=30)
    parser.add_argument("--optimisation_iters", type=int, default=7)
    parser.add_argument("--n_candidates", type=int, default=700)
    parser.add_argument("--top_candidates", type=int, default=70)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--expl_scale", type=float, default=0.01)
    parser.add_argument("--action_noise", type=float, default=0.3)

    # Logging
    parser.add_argument("--sac_log_every", type=int, default=100)
    parser.add_argument("--train_log_every", type=int, default=10)

    # SAC
    parser.add_argument("--sac_discount", default=0.99, type=float)
    parser.add_argument("--sac_init_temperature", default=0.1, type=float)
    parser.add_argument("--sac_alpha_lr", default=1e-4, type=float)
    parser.add_argument("--sac_alpha_beta", default=0.5, type=float)

    # Critic
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)

    # Actor
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)

    # SAC Train
    parser.add_argument("--sac_num_train_steps", default=1000000, type=int)
    parser.add_argument("--sac_batch_size", default=32, type=int)
    parser.add_argument("--sac_hidden_dim", default=1024, type=int)

    #augmented buffer
    parser.add_argument("--n_augments",default=0,type=int)
    parser.add_argument("--augment_std",default=0.01,type=float)
    parser.add_argument("--reward_std",default=0.0,type=float)

    config = parser.parse_args()
    main(config)
