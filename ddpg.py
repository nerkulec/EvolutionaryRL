from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ddpg.core as core
from spinup.utils.logx import EpochLogger
import random
from tqdm import tqdm, trange
from buffer import ReplayBuffer
from evolution import evolution, mutate


def ddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=3e-4, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, max_ep_len=1000, 
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
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    actors = [mutate(deepcopy(ac), mutation_rate) for _ in range(num_actors)]

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    for actor in actors:
        for p in actor.parameters():
            p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    evo_num = 0

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, actor, noise_scale=0):
        a = actor.act(torch.as_tensor(o, dtype=torch.float32))
        if noise_scale != 0:
            a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == env._max_episode_steps)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, ac, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    # env.color = (255, 0, 0)
    # env.render(num_actors=num_actors+1)
    rl_elite_counter = 0
    rl_chosen_counter = 0
    rl_discard_counter = 0
    actor_num = -1
    evo_rewards = []
    # Main loop: collect experience in env and update/log each epoch
    with tqdm(total=epochs) as pbar:
        for t in range(total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise). 
            if t > start_steps:
                if actor_num == -1:
                    a = get_action(o, ac, act_noise)
                else:
                    a = get_action(o, actors[actor_num], 0)
            else:
                a = env.action_space.sample()

            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==env._max_episode_steps else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == env._max_episode_steps):
                if actor_num == -1:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                else:
                    logger.store(EvoEpRet=ep_ret, EvoEpLen=ep_len)
                    evo_rewards.append(ep_ret)
                o, ep_ret, ep_len = env.reset(), 0, 0
                if num_actors > 0:
                    actor_num += 1
                if(actor_num >= len(actors)):
                    evo_num += 1
                    if evo_num % rl_actor_copy_every == 0:
                        actors[-1] = deepcopy(ac)
                        actors[-1]._rl = True
                    actors = evolution(actors, evo_rewards, mutation_rate, num_elites)
                    if evo_num % rl_actor_copy_every == 0:
                        if getattr(actors[0], '_rl', False):
                            rl_elite_counter += 1
                        elif any(getattr(actor, '_rl', False) for actor in actors):
                            rl_chosen_counter += 1
                        else:
                            rl_discard_counter += 1
                        for actor in actors:
                            actor._rl = False
                        s = rl_elite_counter+rl_chosen_counter+rl_discard_counter
                        logger.store(EvoEliteRate=rl_elite_counter/s)
                        logger.store(EvoChosenRate=rl_chosen_counter/s)
                        logger.store(EvoDiscardRate=rl_chosen_counter/s)
                    actor_num = -1
                    evo_rewards = []
                #     env.color = (255, 0, 0)
                # else:
                #     env.color = (0, 0, 255)

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    update(data=batch)
                # env.update_heatmap([ac]+actors)
                # env.render()

            # End of epoch handling
            if (t+1) % steps_per_epoch == 0:
                pbar.update()
                epoch = (t+1) // steps_per_epoch

                # Save model
                if (epoch % save_freq == 0) or (epoch == epochs):
                    logger.save_state({'env': env}, None)

                # Test the performance of the deterministic version of the agent.
                test_agent()

                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                try:
                    logger.log_tabular('EpRet', with_min_and_max=True)
                except:
                    pass
                if num_actors > 0:
                    logger.log_tabular('EvoEpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                try:
                    logger.log_tabular('EpLen', average_only=True)
                except:
                    pass
                if num_actors > 0:
                    logger.log_tabular('EvoEpLen', with_min_and_max=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('QVals', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
                if num_actors > 0:
                    try:
                        logger.log_tabular('EvoEliteRate', average_only=True)
                        logger.log_tabular('EvoChosenRate', average_only=True)
                        logger.log_tabular('EvoDiscardRate', average_only=True)
                    except:
                        pass
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
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--evo_actors', type=int, default=10)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, num_actors=args.evo_actors)
