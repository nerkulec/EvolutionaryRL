from datetime import datetime
import pathlib
from shutil import copy, copytree
import pickle
import json

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

    def save_buffer(self, buffer):
        with open(f'{self.path}/buffer.pkl', 'wb') as f:
            pickle.dump(buffer, f)

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

    def get_buffer_path(self):
        return f'{self.path}/buffer.pkl'

    def __hash__(self):
        return hash((self.env_name, self.alg_name, self.exp_name,
                     hash(frozenset(self.settings.items()))))
