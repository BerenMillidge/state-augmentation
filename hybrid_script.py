import os
import sys
import time
import pathlib
import argparse

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from mbmf.envs import Env
from mbmf.training import Normalizer, Buffer, Trainer
from mbmf.models import EnsembleDynamicsModel
from mbmf.control import RewardMeasure, MpcAgent, RandomSampler, ControlSampler, SacAgent, HybridAgent
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

    normalizer = Normalizer()
    buffer = Buffer(
        state_size,
        action_size,
        config.ensemble_size,
        config.batch_size,
        normalizer=normalizer,
        buffer_size=config.buffer_size,
        device=DEVICE,
    )

    model = EnsembleDynamicsModel(
        state_size + action_size,
        state_size,
        config.ensemble_size,
        config.hidden_size,
        normalizer=normalizer,
        device=DEVICE,
    )

    trainer = Trainer(
        model,
        buffer,
        n_train_epochs=config.n_train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        grad_clip_norm=config.grad_clip_norm,
    )

    reward_measure = RewardMeasure(env, config.reward_scale)
    expl_measure = None
    mpc_agent = MpcAgent(
        model,
        config.ensemble_size,
        action_size,
        plan_horizon=config.plan_horizon,
        optimisation_iters=config.optimisation_iters,
        n_candidates=config.n_candidates,
        top_candidates=config.top_candidates,
        reward_measure=reward_measure,
        expl_measure=expl_measure,
        alpha=config.alpha,
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

    hybrid_agent = HybridAgent(sac_agents, mpc_agent, model, buffer, action_size, n_sac_updates=config.n_sac_updates, cem_std=config.cem_std, device=DEVICE)

    random_sampler = RandomSampler(env)
    hybrid_sampler = ControlSampler(env, hybrid_agent)

    mpc_log_fn = lambda step, reward: print(f"Collect Step {step}: {reward}")
    train_log_fn = lambda epoch, loss: print(f"Train Epoch {epoch}: {loss}")

    random_sampler.sample_record_episodes(config.n_seed_episodes, buffer)
    print(f"Collected {buffer.current_size} seed frames")

    rewards = []
    global_step = 0
    for episode in range(config.n_episodes):
        print(f"\n=== Episode {episode} [{buffer.current_size} frames] ===")

        n_batches = buffer.current_size // config.batch_size
        print(
            f"\nTraining on ({n_batches * config.batch_size}) frames ({n_batches}) batches (buffer size {buffer.current_size})"
        )
        if config.warm_start is 0:
            trainer.reset_models()
        trainer.train(
            n_batches=n_batches, log_fn=train_log_fn, log_every=config.train_log_every
        )

        warm_up = (episode < config.n_warm_up_episodes)
        print(f"\nCollecting {config.n_collect_episodes} episodes of data [warm up: {warm_up}]")
        
        hybrid_agent.toggle_updates(True)
        # TODO double stochastic?
        hybrid_agent.toggle_stochastic(False)
        hybrid_agent.toggle_warm_up(warm_up)
        
        buffer, stats = hybrid_sampler.sample_record_episodes(
            config.n_collect_episodes,
            buffer,
            action_noise=config.action_noise,
            log_fn=mpc_log_fn,
            log_every=config.mpc_log_every,
        )
        print(f"Train reward: {stats['rewards']} Steps: {stats['steps']}")

        if episode % config.test_every == 0:
            print(f"\nTesting on {config.n_test_episodes} episodes")
            hybrid_agent.toggle_updates(False)
            hybrid_agent.toggle_stochastic(False)
            
            stats = hybrid_sampler.sample_episodes(
                config.n_eval_episodes,
                action_noise=None,
                log_fn=mpc_log_fn,
                log_every=config.mpc_log_every
            )
            print(f"Test reward: {stats['rewards']} steps: {stats['steps']}")
            rewards.append(stats["rewards"][0])

    return rewards


if __name__ == "__main__":

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\ndevice [{DEVICE}]")

    parser = argparse.ArgumentParser()

    # Experimental
    parser.add_argument("--n_sac_agents", type=int, default=20)
    # TODO
    parser.add_argument("--n_warm_up_episodes", type=int, default=10)
    parser.add_argument("--cem_std", type=float, default=1.0)
    # TODO
    parser.add_argument("--n_sac_updates", type=int, default=1)
    parser.add_argument("--warm_start", type=int, default=0)
    parser.add_argument("--test_every", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.25)

    # Environment
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--env_name", type=str, default="HalfCheetahRun")
    parser.add_argument("--max_episode_len", type=int, default=100)
    parser.add_argument("--action_repeat", type=int, default=2)

    # Experiment
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_seed_episodes", type=int, default=10)
    parser.add_argument("--n_collect_episodes", type=int, default=1)
    parser.add_argument("--n_test_episodes", type=int, default=1)
    parser.add_argument("--n_eval_episodes", type=int, default=1)

    # Model
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--ensemble_size", type=int, default=5)

    # Training
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--n_train_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--grad_clip_norm", type=float, default=1000)

    # Planning
    parser.add_argument("--plan_horizon", type=int, default=30)
    parser.add_argument("--optimisation_iters", type=int, default=12)
    parser.add_argument("--n_candidates", type=int, default=700)
    parser.add_argument("--top_candidates", type=int, default=70)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--expl_scale", type=float, default=0.01)
    parser.add_argument("--action_noise", type=float, default=0.3)

    # Logging
    parser.add_argument("--mpc_log_every", type=int, default=25)
    parser.add_argument("--train_log_every", type=int, default=100)

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

    config = parser.parse_args()
    main(config)
