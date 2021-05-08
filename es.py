from copy import deepcopy
import itertools
import numpy as np
import torch
import gym
import time
from spinup.utils.logx import EpochLogger
import spinup.algos.pytorch.ddpg.core as core
import random
from tqdm import tqdm, trange
from buffer import ReplayBuffer
from evolution import evolution, mutate


def es(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, gamma=0.99,  
        logger_kwargs=dict(), save_freq=1, num_actors=10, mutation_rate=0.05,
        num_elites=1, rl_actor_copy_every=5):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    actors = [mutate(actor_critic(env.observation_space, env.action_space, **ac_kwargs), mutation_rate)
        for _ in range(num_actors)]

    for actor in actors:
        for p in actor.parameters():
            p.requires_grad = False

    evo_num = 0

    def get_action(o, actor, noise_scale=0):
        a = actor.act(torch.as_tensor(o, dtype=torch.float32))
        if noise_scale != 0:
            a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not(d or (ep_len == env._max_episode_steps)):
            # Take deterministic actions at test time (noise_scale=0)
            o, r, d, _ = test_env.step(get_action(o, actors[0], 0))
            ep_ret += r
            ep_len += 1
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    # env.color = (255, 0, 0)
    # env.render(num_actors=num_actors+1)
    actor_num = 0
    best_evo = -10**6
    evo_rewards = []
    # Main loop: collect experience in env and update/log each epoch
    with tqdm(total=epochs) as pbar:
        for t in range(total_steps):
            a = get_action(o, actors[actor_num], 0)

            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==env._max_episode_steps else d

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == env._max_episode_steps):
                if actor_num == 0 and ep_ret < best_evo:
                    print(f'oops: {ep_ret:5.2f} < {best_evo:5.2f}')
                best_evo = max(best_evo, ep_ret)
                logger.store(EvoEpRet=ep_ret, EvoEpLen=ep_len)
                evo_rewards.append(ep_ret)
                o, ep_ret, ep_len = env.reset(), 0, 0
                actor_num += 1
                if(actor_num >= len(actors)):
                    actors = evolution(actors, evo_rewards, mutation_rate, num_elites)
                    evo_num += 1
                    actor_num = 0
                    evo_rewards = []

            # End of epoch handling
            if (t+1) % steps_per_epoch == 0:
                pbar.update()
                epoch = (t+1) // steps_per_epoch
                test_agent()

                # Save model
                if (epoch % save_freq == 0) or (epoch == epochs):
                    logger.save_state({'env': env}, None)

                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EvoEpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EvoEpLen', with_min_and_max=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='es')
    parser.add_argument('--evo_actors', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    es(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, num_actors=args.evo_actors,
        steps_per_epoch=args.steps_per_epoch)
