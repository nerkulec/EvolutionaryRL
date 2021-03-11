import numpy as np
import random
import gym
import tensorflow as tf
from tqdm.auto import tqdm, trange
import pickle
from collections import defaultdict
import pathlib
from buffer import get_buffer

clone_without_weights = tf.keras.models.clone_model
def clone_with_weights(model):
    new_model = clone_without_weights(model)
    new_model.set_weights([np.copy(w) for w in model.get_weights()])
    return new_model

class ERL:
    def __init__(self, env, actor, critic, optimizer, name = '1', buffer_size = 10**6, actions_per_epoch = 1,
            training_steps_per_epoch = 1, gamma = 0.99, polyak = 0.995, action_noise = (1, 0.1),
            train_summary_writer = None, test_summary_writer = None, num_actors = 10, elite_frac = 0.2,
            episodes_per_actor = 1, episodes_rl_actor = 10, rl_actor_copy_every = 10, mutation_rate = 0.001, mutation_prob = 0.2):
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env
        self.env_id = env.spec.id
        self.actor = actor
        self.critic = critic
        self.target_actor = clone_with_weights(actor)
        self.target_critic = clone_with_weights(critic)
        self.optimizer = optimizer
        self.alg = 'erl'
        self.name = name
        self.buffer = get_buffer(buffer_size, self.env, name = name)

        self.actions_per_epoch = actions_per_epoch
        self.training_steps_per_epoch = training_steps_per_epoch
        self.gamma = gamma
        self.polyak = polyak
        self.train_summary_writer = train_summary_writer
        self.test_summary_writer = test_summary_writer
        self.stats = defaultdict(list) # switch to tensorboard
        self.current_episode_len = 0
        self.cumulative_reward = 0

        # ERL things
        self.actors = [clone_without_weights(self.actor) for _ in range(num_actors)]
        self.elites = int(elite_frac*num_actors)
        self.episodes_per_actor = episodes_per_actor
        self.episodes_rl_actor = episodes_rl_actor
        self.rl_actor_copy_every = rl_actor_copy_every
        self.mutation_rate = mutation_rate
        self.mutation_prob = mutation_prob

        if callable(action_noise):
            self.action_noise = action_noise
        elif type(action_noise) is tuple: # interpolate linearly
            self.action_noise = lambda x: (1-x)*action_noise[0] + x*action_noise[1]
        else:
            self.action_noise = lambda x: action_noise
        

    def train(self, epochs, batch_size = 64, render = False, rl_actor_render = False,
              test_every = 100, render_test = True, render_mode = 'human'):
        self.stats = defaultdict(list)

        if not self.buffer.full():
            self.buffer.fill(self.env)
        
        print("Now training")
        with trange(epochs) as t:
            for epoch in t:
                # Evolutionary actors gather experience
                self.env.color = (0, 0, 255)
                self.env.opacity = 63
                rewards = []
                for actor in self.actors:
                    actor_reward = 0
                    for _ in range(self.episodes_per_actor):
                        state = self.env.reset()
                        done = False
                        while not done:
                            action = self.get_action(actor, state) # 0 noise
                            next_state, reward, done, _ = self.env.step(action)
                            self.buffer.store(state, action, reward, next_state, done)
                            state = next_state
                            actor_reward += reward
                            if render:
                                self.env.render(mode = render_mode)
                    rewards.append(actor_reward)
                self.stats['actors_rewards'].append(list(sorted(rewards)))
                rank = np.argsort(rewards)[::-1] # best on the front
                elites = [clone_with_weights(self.actors[i]) for i in rank[:self.elites]]
                rest = []
                for _ in range(len(self.actors)-self.elites): # tournament selection
                    a = np.random.choice(rank) # select from all
                    b = np.random.choice(rank)
                    if rewards[a] > rewards[b]: # maybe do crossover here
                        winner = self.actors[a]
                    else:
                        winner = self.actors[b]
                    winner = clone_with_weights(winner)
                    if random.random() < self.mutation_prob:
                        winner = self.mutate(winner, self.mutation_rate)
                    rest.append(winner)
                self.actors = elites+rest
                
                # RL actor gathers experience
                self.env.color = (255, 0, 0)
                self.env.opacity = 63
                total_reward = 0
                for _ in range(self.episodes_rl_actor):
                    state = self.env.reset()
                    done = False
                    while not done:
                        action = self.get_action(self.actor, state, self.action_noise(epoch/epochs))
                        next_state, reward, done, _ = self.env.step(action)
                        self.buffer.store(state, action, reward, next_state, done)
                        total_reward += reward
                        state = next_state
                        if render or rl_actor_render:
                            self.env.render(mode = render_mode)
                self.stats['reward'].append(total_reward/self.episodes_rl_actor)
                
                # Train RL actor based on experience
                for _ in range(self.training_steps_per_epoch):
                    self.train_step(batch_size)

                # Copy RL actor into population
                if (epoch+1) % self.rl_actor_copy_every == 0:
                    self.actors[-1] = clone_with_weights(self.actor)
                
                self.env.color = None
                if (epoch+1) % test_every == 0:
                    avg_fitness = self.test(1, render = render_test, render_mode = render_mode)
                    t.set_postfix(test_fitness = avg_fitness)
                    self.test(1, render = render_test, evo_actors = True, only_best = self.elites, render_mode = render_mode)

        return self.stats

    def get_action(self, actor, state, noise=0):
        action = actor(np.array([state])).numpy()[0]
        if noise != 0:
            action = np.random.normal(loc=action, scale=noise) # action noise (change to parameter noise)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def train_step(self, batch_size):
        states, actions, rewards, next_states, done = self.buffer.sample(batch_size)
        self.train_step_critic(states, actions, rewards, next_states, done)
        self.train_step_actor(states)
        self.update_target_networks()

    def train_step_critic(self, states, actions, rewards, next_states, done):
        target_actions = tf.stop_gradient(self.target_actor(next_states))
        target_actions = np.clip(target_actions, self.env.action_space.low, self.env.action_space.high)
        target_critic_input = np.hstack([next_states, target_actions])
        reward_prediction = self.target_critic(target_critic_input)
        targets = tf.stop_gradient(rewards + self.gamma*(1-done)*tf.reshape(reward_prediction, -1))

        critic_input = np.hstack([states, actions])
        
        with tf.GradientTape() as tape:
            critic_ratings = tf.reshape(self.critic(critic_input), -1)
            L = tf.reduce_mean(tf.square(critic_ratings-targets), axis=0) # MSE loss

        self.stats['critic_loss'].append(L.numpy())
        self.stats['critic_pred'].append(critic_ratings.numpy())
        self.stats['critic_weights'].append(self.critic.get_weights()[0])

        gradients = tape.gradient(L, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

    def train_step_actor(self, states):
        with tf.GradientTape() as tape:
            actor_actions = self.actor(states)
            # negative loss so that optimizer maximizes (the critic's rating) instead of minimize
            Q = -tf.reduce_mean(self.critic(tf.concat([states, actor_actions], 1)), axis=0)

        self.stats['actor_loss'].append(Q.numpy())
        self.stats['actor_pred'].append(actor_actions.numpy())
        self.stats['actor_weights'].append(self.actor.get_weights()[0])

        gradients = tape.gradient(Q, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

    def update_target_networks(self):
        def update(model, target_model):
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = self.polyak*target_weights[i] + (1-self.polyak)*weights[i]
            target_model.set_weights(target_weights)
        update(self.critic, self.target_critic)
        update(self.actor, self.target_actor)

    def mutate(self, actor, noise):
        weights = actor.get_weights()
        for i in range(len(weights)):
            weights[i] = np.random.normal(loc=weights[i], scale=noise)
        actor.set_weights(weights)
        return actor

    def test(self, num_episodes = 1, render = True, evo_actors = False, only_best = None, render_mode = 'human'):
        fitness = 0
        if not evo_actors:
            # Testing RLActor
            for _ in range(num_episodes):
                state = self.env.reset()
                done = False
                while not done:
                    action = self.actor(np.array([state])).numpy()[0]
                    state, reward, done, _ = self.env.step(action)
                    if render or (render_mode == 'trajectories' and done):
                        self.env.color = (0, 255, 0)
                        self.env.render(mode = render_mode)
                    fitness += reward
            if self.test_summary_writer:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('reward', fitness/num_episodes)
            self.stats['test_reward'].append(fitness/num_episodes)
            return fitness/num_episodes
        else:
            # Testing EVOActors
            if only_best is None:
                only_best = len(self.actors)
            actors = self.actors[:only_best]
            for actor in actors:
                state = self.env.reset()
                done = False
                while not done:
                    action = actor(np.array([state])).numpy()[0]
                    state, reward, done, _ = self.env.step(action)
                    if render or (render_mode == 'trajectories' and done):
                        self.env.color = (255, 255, 0)
                        self.env.render(mode = render_mode)
                    fitness += reward
            return fitness/len(actors)

    def save_models(self):
        directory = f'models/{self.alg}/{self.env_id}'
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        self.actor.save( f'{directory}/{self.name}-actor')
        self.critic.save(f'{directory}/{self.name}-critic')
