from spinup.utils.run_utils import ExperimentGrid
from ddpg_spinup3 import ddpg
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ddpg-erl-benchmark')
    eg.add('env_name', 'HalfCheetah-v3', '', True)
    eg.add('seed', list(range(args.num_runs)))
    eg.add('epochs', 100)
    eg.add('num_actors', [0, 10])
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(64, 64)], 'hid')
    eg.run(ddpg, num_cpu=args.cpu)