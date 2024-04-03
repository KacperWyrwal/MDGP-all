# Geomstats and GeometricKernels backends 
import geometric_kernels.torch 
import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'


import os 
import torch 
import pandas as pd 
from mdgp.experiments.uci.fit import train, evaluate_deep
from mdgp.experiments.uci.experiment import ExperimentConfigReader, ExperimentConfig
from argparse import ArgumentParser


def run_experiment(experiment_config: ExperimentConfig, device: torch.device, dir_path: str) -> None:
    dataset = experiment_config.data_arguments.dataset
    model = experiment_config.model_arguments.get_model(dataset).to(device)
    projector = experiment_config.model_arguments.get_projector().to(device)
    print("Training...")
    train_loss = train(dataset, model, projector=projector, fit_args=experiment_config.fit_arguments, device=device)
    print("Evaluating...")
    test_metrics = evaluate_deep(dataset, model, projector=projector, fit_args=experiment_config.fit_arguments, device=device)

    print("Saving results...")
    test_metrics_dir = os.path.join(dir_path, 'test')
    os.makedirs(test_metrics_dir, exist_ok=True)
    pd.DataFrame(test_metrics, index=[0]).to_csv(os.path.join(test_metrics_dir, 'metrics.csv'), index=False)

    train_loss_dir = os.path.join(dir_path, 'train')
    os.makedirs(train_loss_dir, exist_ok=True)
    pd.DataFrame({'elbo': train_loss}).to_csv(os.path.join(train_loss_dir, 'metrics.csv'), index=False)
    print("Done.")


def main(dir_path, overwrite=False, config_file_name='config.json'):
    with ExperimentConfigReader(os.path.join(dir_path, config_file_name), overwrite=overwrite) as experiment_config: 
        if experiment_config.can_run:
            experiment_config.set_seed()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            print((
                f"{'Running experiments'.center(80, '=')}\n"
                f"Config: {os.path.join(dir_path, experiment_config.file_name)}\n"
                f"Device: {device}"
            ))
            
            run_experiment(experiment_config=experiment_config, device=device, dir_path=dir_path)
        else:
            print((
                f"Skipping experiement with config: {os.path.join(dir_path, experiment_config.file_name)}"
                f"because it has status: {experiment_config.status}"
            ))


def crawl_and_run(start_directory, config_file_name='config.json', overwrite=False):
    for dirpath, dirnames, filenames in os.walk(start_directory):
        if config_file_name in filenames:
            main(dir_path=dirpath, config_file_name=config_file_name, overwrite=overwrite)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = ArgumentParser(description='Run experiment on UCI datasets')
    parser.add_argument('dir_path', type=str, help='The parent directory to start crawling from.')
    parser.add_argument('--config_name', type=str, default='config.json', help='The name of the config file to match. Default is "config.json".')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing experiments. Default is False.')
    args = parser.parse_args()

    print(f"Looking for experiments to run... (overwrite={args.overwrite})")
    crawl_and_run(start_directory=args.dir_path, config_file_name=args.config_name, overwrite=args.overwrite)
