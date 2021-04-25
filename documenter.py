from datetime import datetime
import pathlib
from shutil import copy, copytree
import pickle
import json
import numpy as np

class Documenter:
    def __init__(self, env_name, alg_name, exp_name, **settings):
        self.env_name = env_name
        self.alg_name = alg_name
        self.exp_name = exp_name
        self.settings = settings
        self.files = []
        self.folders = []
        self.date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        self.path = f'experiments/{self.alg_name}_{self.env_name}/{self.date}_{self.exp_name}'
        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)

    def add_file(self, file_name):
        self.files.append(file_name)
    
    def add_folder(self, folder_name):
        self.folders.append(folder_name)

    def save_files(self):
        pathlib.Path(f'{self.path}/src').mkdir(parents=True, exist_ok=True)
        for file_name in self.files:
            copy(file_name, f'{self.path}/src/{file_name}')
        for folder_name in self.folders:
            copytree(folder_name, f'{self.path}/src/{folder_name}')

    def save_fig(self, plt, plot_name):
        directory = f'{self.path}/plots'
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{directory}/{plot_name}.jpg')

    def save_models(self, alg):
        directory = f'{self.path}/models'
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        alg.actor.save(f'{directory}/actor')
        alg.actor.save(f'{directory}/critic')

    def save_settings(self):        
        with open(f'{self.path}/settings.json', 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=4)

    def get_model_path(self, kind):
        return f'{self.path}/models/{kind}'

    def draw_plots(self, stats):
        import matplotlib.pyplot as plt
        try:
            plt.plot(stats['reward'])
            plt.suptitle('Reward')
            plt.xlabel('Train steps')
            self.save_fig(plt, 'reward')
        except:
            print('Can\'t plot reward')
        # ---
        try:
            plt.plot(stats['critic_loss'])
            plt.suptitle('Critic loss')
            plt.xlabel('Train steps')
            self.save_fig(plt, 'critic_loss')
        except:
            print('Can\'t plot critic_loss')
        # ---
        try:
            plt.plot(stats['actor_loss'])
            plt.suptitle('Actor loss')
            plt.xlabel('Train steps')
            self.save_fig(plt, 'actor_loss')
        except:
            print('Can\'t plot actor_loss')
        # ---
        try:
            critic_pred = np.array(stats['critic_pred'])
            mean = np.mean(critic_pred, axis=1)
            std_dev = np.sqrt(np.mean(np.square(critic_pred-mean.reshape(-1, 1)), axis=1))
            plt.plot(mean)
            plt.fill_between(np.arange(critic_pred.shape[0]), mean-std_dev*1.5, mean+std_dev*1.5, alpha = 0.5)
            plt.suptitle('Critic prediction')
            plt.xlabel('Train steps')
            self.save_fig(plt, 'critic_pred')
        except:
            print('Can\'t plot critic_pred')
        # ---
        try:
            actor_pred = np.array(stats['actor_pred'])
            actor_pred_x = actor_pred[:,:,0]
            actor_pred_y = actor_pred[:,:,1]
            mean = np.mean(actor_pred, axis=1)
            mean_x = mean[:,0]
            mean_y = mean[:,1]
            std_dev_x = np.sqrt(np.mean(np.square(actor_pred_x-mean_x.reshape(-1, 1)), axis=1))
            std_dev_y = np.sqrt(np.mean(np.square(actor_pred_y-mean_y.reshape(-1, 1)), axis=1))
            plt.plot(mean_x)
            plt.plot(mean_y)
            plt.fill_between(np.arange(actor_pred_x.shape[0]), mean_x-std_dev_x*1.5, mean_x+std_dev_x*1.5, alpha = 0.5)
            plt.fill_between(np.arange(actor_pred_y.shape[0]), mean_y-std_dev_y*1.5, mean_y+std_dev_y*1.5, alpha = 0.5)
            plt.suptitle('Actor prediction')
            plt.xlabel('Train steps')
            self.save_fig(plt, 'actor_pred')
        except:
            print('Can\'t plot actor_pred')
        # ---
        try:
            critic_weights = stats['critic_weights']
            fig, ax = plt.subplots()
            for i in range(len(critic_weights[0])):
                color = next(ax._get_lines.prop_cycler)['color']
                plt.plot([w[i].flatten() for w in critic_weights], color = color)
            plt.suptitle('Critic weights')
            plt.xlabel('Train steps')
            self.save_fig(plt, 'critic_weights')
        except:
            print('Can\'t plot critic_weights')
        # ---
        try:
            actor_weights = stats['actor_weights']
            fig, ax = plt.subplots()
            for i in range(len(actor_weights[0])):
                color = next(ax._get_lines.prop_cycler)['color']
                plt.plot([w[i].flatten() for w in actor_weights], color = color)
            plt.suptitle('Actor weights')
            plt.xlabel('Train steps')
            self.save_fig(plt, 'actor_weights')
        except:
            print('Can\'t plot actor_weights')
        # ---
        try:
            actors_rewards = np.array(stats['actors_rewards'])
            mean = np.mean(actors_rewards, axis=1)
            low  = np.min(actors_rewards, axis=1)
            high = np.max(actors_rewards, axis=1)

            plt.plot(actors_rewards)

            plt.suptitle('Evolutionary actors rewards')
            plt.xlabel('Epochs')

            self.save_fig(plt, 'actors_reward')
        except:
            print('Can\'t plot actors_rewards')

