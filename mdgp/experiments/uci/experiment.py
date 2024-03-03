"""
This is almost an exact copy of the mdgp/experiment_utils/reproducibility.py file. 
TODO With some minor tweaks we could make this code more generic use it in both places.
"""
import os 
import json 
import torch 
from dataclasses import dataclass, field, fields, asdict
from itertools import product 
from enum import Enum
from mdgp.experiments.uci.data import DataArguments
from mdgp.experiments.uci.model import ModelArguments
from mdgp.experiments.uci.fit import FitArguments


__all__ = [
    'ExperimentConfig',
    'ExperimentStatus',
    'ExperimentConfigReader',
    'create_experiment_config',
    'create_experiment_config_from_json',
    'set_experiment_seed',
]


# TODO Move to utils
def non_default_fields(dc) -> dict:
    """
    Given a dataclass instance, return a dictionary containing 
    all the fields whose values are different from their defaults.

    Parameters:
    - dc: An instance of a dataclass.

    Returns:
    - Dict[str, Any]: Dictionary with fields and values different from defaults.
    """
    result = {}
    for field in fields(dc):
        current_value = getattr(dc, field.name)        
        if current_value != field.default:
            result[field.name] = current_value
    return result


# TODO Move to utils 
def get_experiment_name(*args) -> str:
    """
    Given an arbitrary number of dataclasses containing the arguments for the experiment,
    return the experiment name based on non-default arguments.
    """
    all_arguments_list = [non_default_fields(arg) for arg in args]
    all_arguments_dict = {k: v for d in all_arguments_list for k, v in d.items()}
    
    if len(all_arguments_dict) == 0: 
        return 'default'
    
    return '-'.join([f'{key}={value}' for key, value in all_arguments_dict.items()])


class ExperimentStatus(Enum): 
    READY = 'ready'
    RUNNING = 'running'
    FAILED = 'failed'
    COMPLETED = 'completed'


@dataclass 
class ExperimentConfig:
    """
    A dataclass that contains the configuration for an experiment. 
    """
    model_arguments: ModelArguments = field(default_factory=ModelArguments, metadata={'help': 'The model arguments.'})
    data_arguments: DataArguments = field(default_factory=DataArguments, metadata={'help': 'The data arguments.'})
    fit_arguments: FitArguments = field(default_factory=FitArguments, metadata={'help': 'The fit arguments.'})
    run: int = field(default=0, metadata={'help': 'The run number of the experiment.'})
    status: ExperimentStatus = field(default=ExperimentStatus.READY, metadata={'help': 'The status of the experiment.'})
    file_name: str = field(default='config.json', metadata={'help': 'The name of the config file.'})
    can_run: bool = field(default=False, init=False, repr=False)

    @property
    def seed(self):
        return self.run
    
    @property 
    def experiment_name(self):
        return get_experiment_name(self.data_arguments, self.fit_arguments, self.model_arguments)
    
    @property 
    def run_name(self):
        return f"run_{self.run}"
    
    def set_seed(self) -> None:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
    
    def to_dict(self): 
        return {
            'model_arguments': asdict(self.model_arguments),
            'data_arguments': asdict(self.data_arguments),
            'fit_arguments': asdict(self.fit_arguments),
            'run': self.run,
            'status': self.status.value,
        }
    
    def to_json(self, dir_path):
        file_path = os.path.join(dir_path, self.file_name)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data):
        data['status'] = ExperimentStatus(data['status'])
        data['model_arguments'] = ModelArguments(**data['model_arguments'])
        data['data_arguments'] = DataArguments(**data['data_arguments'])
        data['fit_arguments'] = FitArguments(**data['fit_arguments'])
        data['run'] = data.get('run', 0)
        return cls(**data)

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            data['file_name'] = os.path.basename(file_path)
        return cls.from_dict(data)


# TODO Move to utils
def expand_args(args_dict):
    """
    Converts a dictionary of arguments into a list of dictionaries where each dictionary is a unique 
    combination of arguments.
    """
    keys = args_dict.keys()
    values = (args_dict[key] if isinstance(args_dict[key], list) else [args_dict[key]] for key in keys)
    return [dict(zip(keys, combination)) for combination in product(*values)]


def create_experiment_config_from_json(json_config, dir_path, overwrite=False) -> None:
    """
    Creates a folder for each experiment and a config file for each run. 

    Parameters:
    - json_config (dict): A dictionary containing the experiment configurations.
    - dir_path (str): The path to the parent directory where the experiment folders will be created.
    """

    # Expand arguments to get all possible configurations
    model_args_list = expand_args(json_config['model_arguments'])
    data_args_list = expand_args(json_config['data_arguments'])
    fit_arguments = expand_args(json_config['fit_arguments'])
    runs = json_config['runs']

    # Get all combinations of arguments
    for model_args, data_args, fit_args in product(model_args_list, data_args_list, fit_arguments):
        for run in runs:
            config = ExperimentConfig(
                model_arguments=ModelArguments(**model_args),
                data_arguments=DataArguments(**data_args),
                fit_arguments=FitArguments(**fit_args),
                run=run,
            )

            # Create experiment directory if it doesn't exist
            experiment_path = os.path.join(dir_path, config.experiment_name)
            os.makedirs(experiment_path, exist_ok=True)

            # Create run directory if doesn't exist 
            run_path = os.path.join(experiment_path, config.run_name)
            os.makedirs(run_path, exist_ok=True)

            # Maybe skip if config file already exists
            config_path = os.path.join(run_path, config.file_name)
            if os.path.exists(config_path) and not overwrite: 
                print(f"Skipping config file {config_path} because it already exists and overwrite is set to False.")
                continue 
            
            # Save config file 
            config.to_json(run_path)


def create_experiment_config(json_config_path, dir_path, overwrite=False) -> None: 
    """
    Creates a folder for each experiment and a config file for each run. 

    Parameters:
    - dir_path (str): The path to the parent directory where the experiment folders will be created.
    - json_config_path (str): The path to the JSON file containing the experiment configurations.
    """

    # Read the JSON file
    with open(json_config_path, 'r') as f:
        json_config = json.load(f)
    
    create_experiment_config_from_json(json_config, dir_path, overwrite=overwrite)


class ExperimentConfigReader:
    def __init__(self, file_path, overwrite=False):
        self.file_path = file_path
        self.overwrite = overwrite
        self.experiment_config = None

    def __enter__(self):
        # This part is executed when entering the 'with' block
        self.experiment_config = ExperimentConfig.from_json(self.file_path)
        if (self.experiment_config.status == ExperimentStatus.READY or 
            self.experiment_config.status == ExperimentStatus.FAILED or 
            (self.experiment_config.status == ExperimentStatus.COMPLETED and self.overwrite)):
            self._update_status(ExperimentStatus.RUNNING)
            self.experiment_config.can_run = True 
        return self.experiment_config

    def __exit__(self, exc_type, exc_value, traceback): 
        if self.experiment_config.can_run is True:
            status = ExperimentStatus.COMPLETED if exc_type is None else ExperimentStatus.FAILED
            self._update_status(status)
            # This is kind of redundant, since the context manager will no longer be used, 
            # but it makes it more explicit
            self.experiment_config.can_run = False 

    def _update_status(self, status):
        # Instead of directly modifying the JSON, we update the ExperimentConfig object and save it
        self.experiment_config.status = status
        self.experiment_config.to_json(os.path.dirname(self.file_path))
