from argparse import ArgumentParser
from mdgp.experiment_utils import create_experiment_config    


if __name__ == '__main__':
    parser = ArgumentParser(description='Create experiment config files')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing config files')
    parser.add_argument('--dir_path', type=str, default='../../../experiments/benchmark', help='The directory to save the config files to')
    parser.add_argument('--config_path', type=str, default='../../../experiment_tree_configs/euclidean_dgp.json', help='The path to the config file to create')
    args = parser.parse_args()

    create_experiment_config(json_config_path=args.config_path, dir_path=args.dir_path, overwrite=args.overwrite)