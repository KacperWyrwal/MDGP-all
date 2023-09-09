from torch import no_grad 
from dataclasses import dataclass, field
from tqdm.autonotebook import tqdm 
from gpytorch.metrics import (
    mean_absolute_error, mean_squared_error, mean_standardized_log_loss, 
    standardized_mean_squared_error, quantile_coverage_error, negative_log_predictive_density
)
from mdgp.experiment_utils.logging import log 


__all__ = [
    'TrainingArguments',
    'train_step',
    'test_step',
    'fit',
]


@dataclass
class TrainingArguments: 
    num_steps: int = field(default=1000, metadata={'help': 'Number of steps to train for'})
    val_every_n_steps: int = field(default=50, metadata={'help': 'Number of steps between validation runs'})
    sample_hidden: str = field(default='naive', metadata={'help': 'Name of the function to sample from the hidden space. Must be one of ["naive", "pathwise"]'})


def train_step(model, inputs, targets, criterion, sample_hidden='naive', loggers=None, step=None): 
    model.train() 
    outputs = model(inputs, sample_hidden=sample_hidden)
    loss = criterion(outputs, targets)
    log(loggers=loggers, metrics={'elbo': loss}, step=step)
    return loss 


def test_step(model, inputs, targets, sample_hidden='naive', train_targets=None, loggers=None, step=None):
    with no_grad():
        model.eval() 
        outputs_f = model(inputs, sample_hidden=sample_hidden)
        outputs_y = model.likelihood(outputs_f)
        metrics = {
            'expected_log_probability': model.likelihood.expected_log_prob(targets, outputs_f).mean(), 
            'mean_absolute_error': mean_absolute_error(outputs_y, targets).mean(0), 
            'mean_squared_error': mean_squared_error(outputs_y, targets).mean(0), 
            'standardized_mean_squared_error': standardized_mean_squared_error(outputs_y, targets).mean(0), 
            'mean_standardized_log_loss': mean_standardized_log_loss(outputs_y, targets, train_y=train_targets).mean(0), 
            'quantile_coverage_error': quantile_coverage_error(outputs_y, targets).mean(0), 
            'negative_log_predictive_density': negative_log_predictive_density(outputs_y, targets).mean(0)
        }
    log(loggers=loggers, metrics=metrics, step=step)
    return metrics 


def fit(model, optimizer, criterion, train_inputs, train_targets, val_inputs=None, val_targets=None, train_loggers=None, val_loggers=None, training_args: TrainingArguments = None): 
    validate = val_inputs is not None and val_targets is not None
    metrics = {'elbo': None, 'nlpd': None, 'smse': None}

    pbar = tqdm(range(1, training_args.num_steps + 1), desc="Fitting")
    for step in pbar:
        # Training step and display training metrics 
        optimizer.zero_grad(set_to_none=True)
        loss = train_step(model=model, inputs=train_inputs, targets=train_targets, criterion=criterion, sample_hidden=training_args.sample_hidden, 
                          loggers=train_loggers, step=step)
        loss.backward()
        optimizer.step() 
        metrics.update({'elbo': loss.item()})

        # Validation step and display validation metrics 
        if validate and ((step - 1) % training_args.val_every_n_steps == 0):
            val_metrics = test_step(model=model, inputs=val_inputs, targets=val_targets, sample_hidden=training_args.sample_hidden, 
                                    train_targets=train_targets, loggers=val_loggers, step=step)
            metrics.update({'nlpd': val_metrics['negative_log_predictive_density'].item(), 
                            'smse': val_metrics['standardized_mean_squared_error'].item()})
        # Display metrics 
        pbar.set_postfix(metrics)
    return model 
    