import geometric_kernels.torch
import os 
import warnings 
import torch 
from torch import set_default_dtype, float64
from mdgp.bo_experiment.data import get_initial_data
from mdgp.bo_experiment.model import create_model
from mdgp.experiment_utils.logging import CSVLogger, finalize 
from mdgp.bo_experiment.fit import fit
from mdgp.bo_experiment import ExperimentConfig, ExperimentConfigReader, set_experiment_seed
from mdgp.bo_experiment import BOArguments, ModelArguments, FitArguments, optimize_acqf_manifold
from mdgp.experiment_utils import log, finalize 
from tqdm.autonotebook import tqdm 
from botorch.fit import fit_gpytorch_mll
from argparse import ArgumentParser


def run_bo(initial_data, target_function, bo_args: BOArguments, model_args: ModelArguments, fit_args: FitArguments, loggers=None, show_fit_progress=False):
    warnings.filterwarnings("ignore")

    x = initial_data
    y = target_function(initial_data)
    best_x = initial_data[y.argmin()]
    best_y = y.min()
         
    pbar = tqdm(range(bo_args.num_iter), desc="BO")
    for _ in pbar: 
        # 1. Create model, mll, and optimizer 
        model = create_model(model_args=model_args, train_x=x, train_y=y, inducing_points=x)
        optimizer = model_args.optimizer_factory(model=model.base_model, lr=fit_args.lr)
        mll = model_args.mll_factory(model.base_model, y=y)

        # 2. Fit model to observations  
        if model_args.model_name == 'exact':
            fit_gpytorch_mll(mll=mll)
        elif model_args.model_name == 'deep':
            fit(model=model, optimizer=optimizer, criterion=mll, train_inputs=x, train_targets=y, fit_args=fit_args, show_progress=show_fit_progress)
        else:
            raise ValueError(f"Unknown model name: {model_args.model_name}")

        # 4. Get acquisition function for the fitted model 
        acq_function = model_args.acqf_factory(model=model, best_f=best_y)

        # 5. Get new observation 
        with torch.no_grad():
            new_x, _ = optimize_acqf_manifold(acq_function=acq_function, bo_args=bo_args)

        # 6. Observe target function at acquired point and add to previous observations 
        new_x = new_x.unsqueeze(-2)
        new_y = target_function(new_x).squeeze(0)

        print(new_x, new_y)

        x = torch.cat([x, new_x])
        y = torch.cat([y, new_y])

        # 7. Update best observation
        if new_y < best_y: 
            best_y = new_y 
            best_x = new_x.squeeze()

        # 8. Log best observation
        metrics = dict(
            best_x=best_x.tolist(), 
            best_y=best_y.item(),
        )
        log(loggers=loggers, metrics=metrics)
        pbar.set_postfix(metrics)


def run_experiment(experiment_config: ExperimentConfig, dir_path: str, show_fit_progress: bool = False):
    print(f"Running experiment with the config: {os.path.join(dir_path, experiment_config.file_name)}")
    # 0. Unpack arguments
    model_args, data_args, fit_args, bo_args = (
        experiment_config.model_arguments, experiment_config.data_arguments, experiment_config.fit_arguments, experiment_config.bo_arguments
    )

    # 1. Get initial data and target function. Target function is observed at input points acquired via BO 
    print("Creating initial observations..")
    target_function = data_args.target_function
    initial_data = get_initial_data(data_args=data_args)

    # 2. Set up logger for capturing points and observations acquired via BO
    bo_loggers = [CSVLogger(root_dir=os.path.join(dir_path, 'bo'), flush_logs_every_n_steps=5)]

    # 3. Run BO loop
    print("Running Bayesian optimisation..")
    run_bo(initial_data=initial_data, target_function=target_function, bo_args=bo_args, model_args=model_args, fit_args=fit_args, loggers=bo_loggers, show_fit_progress=show_fit_progress)
    finalize(bo_loggers)
    
    print("Done!")


# TODO main, crawl_are_run, and __main__ are generic enough to be extracted into a separate module and used in 
# run.py for every experiment. 


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
    parser.add_argument('--overwrite', type=bool, default=False, help='Whether to overwrite existing experiments. Default is False.')
    args = parser.parse_args()
    
    crawl_and_run(start_directory=args.dir_path, config_file_name=args.config_name, overwrite=args.overwrite)
