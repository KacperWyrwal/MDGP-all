from dataclasses import dataclass, field
from tqdm.autonotebook import tqdm 
from mdgp.experiment_utils.logging import log 



@dataclass
class FitArguments: 
    num_steps: int = field(default=500, metadata={'help': 'Number of steps to train for'})
    sample_hidden: str = field(default='naive', metadata={'help': 'Name of the function to sample from the hidden space. Must be one of ["naive", "pathwise"]'})
    lr: float = field(default=1e-2, metadata={'help': 'Learning rate'})


def train_step(model, inputs, targets, criterion, sample_hidden='naive', loggers=None, step=None): 
    model.train() 
    outputs = model(inputs, sample_hidden=sample_hidden)
    loss = criterion(outputs, targets)
    log(loggers=loggers, metrics={'elbo': loss}, step=step)
    return loss 


def fit(model, optimizer, criterion, train_inputs, train_targets, train_loggers=None, fit_args: FitArguments = None, show_progress=True): 
    metrics = {'elbo': None}
    pbar = tqdm(range(1, fit_args.num_steps + 1), desc="Fitting", leave=False, disable=not show_progress)
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
    