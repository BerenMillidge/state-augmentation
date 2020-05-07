""" 

The purpose of this script is to test whether a fully trained SAC + ensemble model work with MPC
We test whether an ensemble of SAC agents is useful

__author__: Alexander Tschantz
__date__: 02/05/2020

"""

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
from mbmf.control import RewardMeasure, MpcAgent, EnsembleSacAgent, RandomSampler, ControlSampler, SacAgent, HybridAgent
from mbmf import utils

def make_sac_agent(state_shape, action_shape, args, device):
    """ TODO move to some other file """
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


def train_sac_and_model(args, L):
    """ Trains SAC agents and ensemble model 
        Data is collected via SAC agent 
    """
    L.log(f"\n\n== Training SAC agents and ensemble model == \n\n")
    utils.seed(args.seed)

    # Environment
    env = Env(
        args.env_name,
        max_episode_len=args.max_episode_len,
        action_repeat=args.action_repeat,
        seed=args.seed,
    )
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # Buffer
    normalizer = Normalizer()
    buffer = Buffer(
        state_size,
        action_size,
        args.ensemble_size,
        args.batch_size,
        normalizer=normalizer,
        buffer_size=args.buffer_size,
        device=DEVICE,
    )

    # Model
    model = EnsembleDynamicsModel(
        state_size + action_size,
        state_size,
        args.ensemble_size,
        args.hidden_size,
        normalizer=normalizer,
        device=DEVICE,
    )

    trainer = Trainer(
        model,
        buffer,
        n_train_epochs=args.n_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        grad_clip_norm=args.grad_clip_norm,
    )

    # SAC agents
    sac_agents = [
        make_sac_agent(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=DEVICE,
        )
        for _ in range(args.n_sac_agents)
    ]
    sac_agent = EnsembleSacAgent(sac_agents, buffer)

    # Data sampler
    random_sampler = RandomSampler(env)
    sac_sampler = ControlSampler(env, sac_agent)

    # Logging
    sac_log_fn = lambda step, reward: L.log(f"Collect Step {step}: {reward}")
    train_log_fn = lambda epoch, loss: L.log(f"Train Epoch {epoch}: {loss}")

    # Collect random data
    random_sampler.sample_record_episodes(args.n_seed_episodes, buffer)
    L.log(f"Collected {buffer.current_size} seed frames")

    # Main loop
    rewards = []
    global_step = 0
    for episode in range(args.n_train_epi):
        L.log(f"\nEpisode {episode} [{buffer.current_size} frames]")

        # Collect data
        L.log(f"Collecting {args.n_collect_episodes} episodes of data")
        
        sac_agent.toggle_updates(True)
        sac_agent.toggle_stochastic(True)

        buffer, stats = sac_sampler.sample_record_episodes(
            args.n_collect_episodes,
            buffer,
            action_noise=None,
            log_fn=sac_log_fn,
            log_every=args.mpc_log_every,
        )
        L.log_episode(stats['rewards'][0], stats['steps'][0])
        L.save()

    # Train Model
    n_batches = buffer.current_size // args.batch_size
    L.log(
        f"\nTraining on ({n_batches * args.batch_size}) frames ({n_batches}) batches | buffer size ({buffer.current_size})\n"
    )
    trainer.train(
        n_batches=n_batches, log_fn=train_log_fn, log_every=args.train_log_every
    )
    sac_agent.save(L.path)
    trainer.save_models(L.path)


    return sac_agent, model


def test_sac_and_model(sac_agent, model, args, L):
    """ Tests the trained SAC agents and ensemble model with the MPC planner (i.e. `hybrid` algo)
        No training takes place
    """
    L.log(f"\n\n== Testing trained SAC and ensemble model with MPC planner  == \n\n")
    utils.seed(args.seed)

    # Environment
    env = Env(
        args.env_name,
        max_episode_len=args.max_episode_len,
        action_repeat=args.action_repeat,
        seed=args.seed,
    )
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # MPC Planner 
    reward_measure = RewardMeasure(env, args.reward_scale)
    expl_measure = None
    mpc_agent = MpcAgent(
        model,
        args.ensemble_size,
        action_size,
        plan_horizon=args.plan_horizon,
        optimisation_iters=args.optimisation_iters,
        n_candidates=args.n_candidates,
        top_candidates=args.top_candidates,
        reward_measure=reward_measure,
        expl_measure=expl_measure,
        device=DEVICE,
    )

    # hybrid agent 
    sac_agents = sac_agent.sac_agents
    hybrid_agent = HybridAgent(sac_agents, mpc_agent, model, None, action_size, L=L, n_sac_updates=args.n_sac_updates, device=DEVICE)

    # Data samplers
    random_sampler = RandomSampler(env)
    hybrid_sampler = ControlSampler(env, hybrid_agent)

    # Logging
    mpc_log_fn = lambda step, reward: L.log(f"Collect Step {step}: {reward}")
    train_log_fn = lambda epoch, loss: L.log(f"Train Epoch {epoch}: {loss}")

    # Main loop
    rewards = []
    global_step = 0
    for episode in range(args.n_test_epi):
        L.log(f"\nEpisode {episode}")

        # Test agent 
        L.log(f"Testing on {args.n_test_episodes} episodes")
        
        sac_agent.toggle_updates(False)
        sac_agent.toggle_stochastic(False)
        
        stats = hybrid_sampler.sample_episodes(
            args.n_eval_episodes,
            action_noise=None,
            log_fn=mpc_log_fn,
            log_every=args.mpc_log_every
        )
        L.log_episode(stats['rewards'], stats['steps'])
        L.save()

    return rewards


def test_mpc_and_model(model, args, L):
    """ Tests the model (is MPC helping?)
    """
    L.log(f"\n\n== Testing trained ensemble model with MPC planner  == \n\n")
    utils.seed(args.seed)

    # Environment
    env = Env(
        args.env_name,
        max_episode_len=args.max_episode_len,
        action_repeat=args.action_repeat,
        seed=args.seed,
    )
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # MPC Planner 
    reward_measure = RewardMeasure(env, args.reward_scale)
    expl_measure = None
    mpc_agent = MpcAgent(
        model,
        args.ensemble_size,
        action_size,
        plan_horizon=args.plan_horizon,
        optimisation_iters=args.optimisation_iters,
        n_candidates=args.n_candidates,
        top_candidates=args.top_candidates,
        reward_measure=reward_measure,
        expl_measure=expl_measure,
        device=DEVICE,
    )

    # Data samplers
    random_sampler = RandomSampler(env)
    hybrid_sampler = ControlSampler(env, mpc_agent)

    # Logging
    mpc_log_fn = lambda step, reward: L.log(f"Collect Step {step}: {reward}")
    train_log_fn = lambda epoch, loss: L.log(f"Train Epoch {epoch}: {loss}")

    # Main loop
    rewards = []
    global_step = 0
    for episode in range(args.n_test_epi):
        L.log(f"\nEpisode {episode}")

        # Test agent 
        L.log(f"Testing on {args.n_test_episodes} episodes")
        
        sac_agent.toggle_updates(False)
        sac_agent.toggle_stochastic(False)
        
        stats = hybrid_sampler.sample_episodes(
            args.n_eval_episodes,
            action_noise=None,
            log_fn=mpc_log_fn,
            log_every=args.mpc_log_every
        )
        L.log_episode(stats['rewards'], stats['steps'])
        L.save()

    return rewards


def main(args):
    L = utils.Logger(args.logdir, args.seed)
    sac_agent, model = train_sac_and_model(args, L)
    test_sac_and_model(sac_agent, model, args, L)
    test_mpc_and_model(model, args, L)


if __name__ == "__main__":

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser()

    # Experimental
    parser.add_argument("--n_sac_agents", type=int, default=5)
    parser.add_argument("--n_sac_updates", type=int, default=1)
    parser.add_argument("--warm_start", type=int, default=0)

    # Environment
    parser.add_argument("--logdir", type=str, default="train_exp_ensemble")
    parser.add_argument("--env_name", type=str, default="HalfCheetahRun")
    parser.add_argument("--max_episode_len", type=int, default=100)
    parser.add_argument("--action_repeat", type=int, default=2)

    # Experiment
    parser.add_argument("--n_train_epi", type=int, default=1000)
    parser.add_argument("--n_test_epi", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--optimisation_iters", type=int, default=7)
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

    args = parser.parse_args()

    main(args)
