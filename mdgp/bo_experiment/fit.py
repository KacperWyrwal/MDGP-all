from dataclasses import dataclass, field
from tqdm.autonotebook import tqdm 
from mdgp.experiment_utils.logging import log 
from mdgp.bo_experiment.utils import ExcludeFromNameMixin



@dataclass
class FitArguments(ExcludeFromNameMixin): 
    num_steps: int = field(default=500, metadata={'help': 'Number of steps to train for'})
    sample_hidden: str = field(default='naive', metadata={'help': 'Name of the function to sample from the hidden space. Must be one of ["naive", "pathwise"]'})
    lr: float = field(default=1e-2, metadata={'help': 'Learning rate'})
    full_every_n_steps: int = field(default=1, metadata={'help': 'Number of steps between full updates'})
    partial_num_steps_ratio: float = field(default=0.1, metadata={'help': 'Ratio of steps to use for partial fit'})

    @property
    def partial_num_steps(self):
        return int(self.num_steps * self.partial_num_steps_ratio)


def train_step(model, inputs, targets, criterion, sample_hidden='naive', loggers=None, step=None): 
    model.train() 
    outputs = model(inputs, sample_hidden=sample_hidden)
    loss = criterion(outputs, targets)
    log(loggers=loggers, metrics={'elbo': loss}, step=step)
    return loss 


def fit(model, optimizer, criterion, train_inputs, train_targets, train_loggers=None, fit_args: FitArguments = None, show_progress=True, num_steps=None): 
    metrics = {'elbo': None}
    pbar = tqdm(range(1, (num_steps or fit_args.num_steps) + 1), desc="Fitting", leave=False, disable=not show_progress)
    for step in pbar:
        # Training step and display training metrics 
        optimizer.zero_grad(set_to_none=True)
        loss = train_step(model=model, inputs=train_inputs, targets=train_targets, criterion=criterion, sample_hidden=fit_args.sample_hidden, 
                          loggers=train_loggers, step=step)
        loss.backward()
        optimizer.step() 
        metrics.update({'elbo': loss.item()})
        # Display metrics 
        pbar.set_postfix(metrics)
    return model 
    