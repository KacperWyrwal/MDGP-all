from torch import Tensor


import torch 
import gpytorch 
from torch.utils.data.dataloader import DataLoader, default_collate
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from gpytorch.metrics import negative_log_predictive_density
from gpytorch.models.deep_gps import DeepGP
from tqdm.auto import tqdm
from math import ceil
from mdgp.experiments.uci.data.datasets import UCIDataset
from mdgp.experiments.uci.fit.metrics import test_log_likelihood, mean_squared_error, negative_log_predictive_density, test_log_likelihood_batch, squared_error_batch
from dataclasses import dataclass, field
from mdgp.experiments.uci.model.projectors import Projector 


# Settings from the paper 
DEFAULT_BATCH_SIZE = 10_000
NUM_ITERATIONS = 20_000
LR = 0.01


def _get_batch_size(dataset: UCIDataset, batch_size: int) -> int:
    return min(batch_size, dataset.train_x.size(0))


@dataclass
class FitArguments: 

    train_batch_size: int = field(default=DEFAULT_BATCH_SIZE, metadata={'help': 'Batch size for training.'})
    test_batch_size: int | None = field(default=None, metadata={'help': 'Batch size for testing.'})
    num_iterations: int = field(default=NUM_ITERATIONS, metadata={'help': 'Number of iterations to train the model for.'})
    train_num_samples: int = field(default=10)
    test_num_samples: int = field(default=10)

    def __post_init__(self):
        if self.test_batch_size is None:
            self.test_batch_size = self.train_batch_size

    def get_train_batch_size(self, dataset: UCIDataset) -> int:
        return _get_batch_size(dataset, self.train_batch_size)
    
    def get_test_batch_size(self, dataset: UCIDataset) -> int:
        return _get_batch_size(dataset, self.test_batch_size)

    def num_epochs(self, dataset: UCIDataset) -> int:
        default_iterations_per_epoch = ceil(dataset.train_x.size(0) / _get_batch_size(dataset, DEFAULT_BATCH_SIZE))
        return ceil(self.num_iterations / default_iterations_per_epoch)


def to_device(collate_fn, device: torch.device):
    def to_device_fn(x: Tensor) -> Tensor:
        return tuple(_x.to(device) for _x in collate_fn(x))
    return to_device_fn


def train_step(x: Tensor, y: Tensor, model: DeepGP, optimizer: torch.optim.Optimizer, elbo: VariationalELBO) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = elbo(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(dataset: UCIDataset, model: DeepGP, projector: Projector, fit_args: FitArguments, device: torch.device, inner_pbar: bool = False) -> list[float]: 
    with gpytorch.settings.num_likelihood_samples(fit_args.train_num_samples):
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, maximize=True)
        elbo = DeepApproximateMLL(VariationalELBO(model.likelihood, model, dataset.train_y.size(0)))
        train_loader = DataLoader(dataset.train_dataset, batch_size=fit_args.get_train_batch_size(dataset), 
                                shuffle=True, collate_fn=to_device(default_collate, device))

        losses = []
        for _ in (pbar := tqdm(range(fit_args.num_epochs(dataset)), desc='Epochs', position=0, leave=True)):
            epoch_loss = 0
            batches_pbar = tqdm(train_loader, total=len(train_loader), desc="Batches", position=1, leave=True)
            for x_batch, y_batch in batches_pbar:
                x_batch, y_batch = projector(x_batch, y_batch)
                loss = train_step(x=x_batch, y=y_batch, model=model, optimizer=optimizer, elbo=elbo)
                epoch_loss += loss
            losses.append(epoch_loss)
            pbar.set_postfix({'ELBO': epoch_loss})

        return losses 


def evaluate_deep(dataset: UCIDataset, model: DeepGP, projector: Projector, device: torch.device, fit_args: FitArguments) -> dict[str, float]:
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(fit_args.test_num_samples):
        test_y_std = dataset.test_y_std.to(device)
        test_loader = DataLoader(dataset.test_dataset, batch_size=fit_args.get_test_batch_size(dataset), 
                                 shuffle=True, collate_fn=to_device(default_collate, device))
        tll, se = [], []
        for x_batch, y_batch in test_loader:
            x_batch = projector(x_batch)
            out = model.likelihood(model(x_batch))
            out = projector.inverse(out)
            tll.append(test_log_likelihood_batch(out, y_batch, test_y_std))
            se.append(squared_error_batch(out, y_batch, test_y_std))
        tll = torch.cat(tll, dim=-1).mean().item()
        mse = torch.cat(se, dim=-1).mean().item()
        print(f"TLL: {tll}, MSE: {mse}")
    return {'tll': tll, 'mse': mse}

        
def evaluate_shallow(dataset: UCIDataset, model: DeepGP, projector: Projector, device: torch.device, fit_args: FitArguments | None = None) -> dict[str, float]:
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
        test_x = dataset.test_x.to(device)
        test_y = dataset.test_y.to(device)
        test_y_std = dataset.test_y_std.to(device)

        test_x = projector(test_x)
        out = model.likelihood(model(test_x))
        out = projector.inverse(out)
        tll = test_log_likelihood(out, test_y, test_y_std)
        mse = mean_squared_error(out, test_y, test_y_std)
        nlpd = negative_log_predictive_density(out, test_y)
        metrics = {
            'tll': tll.mean().item(), 
            'mse': mse.mean().item(),
            'nlpd': nlpd.mean().item(),
        }
        print(f"TLL: {metrics['tll']}, MSE: {metrics['mse']}")
    return metrics 
