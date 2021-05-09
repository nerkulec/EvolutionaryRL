from spinup.utils.run_utils import ExperimentGrid
from ddpg import ddpg
from td3 import td3
from sac import sac
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ddpg-erl-benchmark-recreate-paper')
    eg.add('env_name', 'Swimmer-v3', '', True)
    eg.add('seed', list(range(args.num_runs)))
    eg.add('epochs', 500)
    eg.add('num_actors', [10])
    eg.add('steps_per_epoch', 11000)
    eg.add('ac_kwargs:hidden_sizes', [(256, 256)], 'hid')
    eg.run(ddpg, num_cpu=args.cpu)