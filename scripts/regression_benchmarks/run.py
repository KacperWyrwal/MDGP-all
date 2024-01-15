import geometric_kernels.torch
import os 
from argparse import ArgumentParser
from torch import set_default_dtype, float64
from torch.optim import Adam  
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO, ExactMarginalLogLikelihood
from mdgp.experiments.experiment_utils.data import get_data 
from mdgp.experiments.experiment_utils.model import create_model
from mdgp.experiments.experiment_utils.logging import CSVLogger, finalize 
from mdgp.experiments.experiment_utils.training import fit, test_step
from mdgp.experiments.experiment_utils import ExperimentConfigReader, set_experiment_seed


def run_experiment(experiment_config, dir_path):
    print(f"Running experiment with the config: {os.path.join(dir_path, experiment_config.file_name)}")
    # 0. Unpack arguments
    model_args, data_args, training_args = experiment_config.model_arguments, experiment_config.data_arguments, experiment_config.training_arguments

    # 1. Get data 
    train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = get_data(data_args=data_args)

    # 2. Create model, criterion, and optimizer 
    model = create_model(model_args=model_args, train_x=train_inputs, train_y=train_targets)
    if model_args.model_name == 'exact':
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    else:
        mll = DeepApproximateMLL(
            VariationalELBO(likelihood=model.likelihood, model=model, num_data=data_args.num_train)
        )
    optimizer = Adam(model.parameters(), maximize=True, lr=0.01) # Maximize because we are working with ELBO not negative ELBO 
    
    # 4. Train and validate model
    print("Training...")
    train_csv_logger = CSVLogger(root_dir=os.path.join(dir_path, 'train')) 
    val_csv_logger = CSVLogger(root_dir=os.path.join(dir_path, 'val')) 
    train_loggers = [train_csv_logger]
    val_loggers = [val_csv_logger]
    model = fit(model=model, optimizer=optimizer, criterion=mll, train_loggers=train_loggers, 
                val_loggers=val_loggers, train_inputs=train_inputs, train_targets=train_targets,
                val_inputs=val_inputs, val_targets=val_targets, training_args=training_args)

    # make sure logger files are saved
    finalize(loggers=[*val_loggers, *train_loggers])

    # # 5. Test model 
    print("Testing...")
    test_csv_logger = CSVLogger(root_dir=os.path.join(dir_path, 'test'))
    test_loggers = [test_csv_logger]
    test_metrics = test_step(model=model, inputs=test_inputs, targets=test_targets, sample_hidden=training_args.sample_hidden, 
                             loggers=test_loggers, train_targets=train_targets)
    # make sure logger files are saved
    finalize(loggers=test_loggers)
    print(test_metrics)
    print("Done!")


def main(dir_path, overwrite=False, config_file_name='config.json'):
    with ExperimentConfigReader(os.path.join(dir_path, config_file_name), overwrite=overwrite) as experiment_config: 
        if experiment_config.can_run:
            set_experiment_seed(experiment_config.seed)
            run_experiment(experiment_config=experiment_config, dir_path=dir_path)
        else:
            print(f"""Skipping expeiement with the config: {os.path.join(dir_path, experiment_config.file_name)}
            because it has status: {experiment_config.status}""")


def crawl_and_run(start_directory, config_file_name='config.json', overwrite=False):
    # Iterate through the directory tree starting from the given directory
    for dirpath, dirnames, filenames in os.walk(start_directory):
        # Check if the file_name_to_match is in the current directory's file list
        if config_file_name in filenames:
            main(dir_path=dirpath, config_file_name=config_file_name, overwrite=overwrite)


if __name__ == "__main__":
    set_default_dtype(float64)

    # Parse arguments 
    parser = ArgumentParser(description='Crawl through directories and run experiments based on config files.')
    parser.add_argument('dir_path', type=str, help='The parent directory to start crawling from.')
    parser.add_argument('--config_name', type=str, default='config.json', help='The name of the config file to match. Default is "config.json".')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing experiments. Default is False.')
    args = parser.parse_args()
    
    crawl_and_run(start_directory=args.dir_path, config_file_name=args.config_name, overwrite=args.overwrite)

