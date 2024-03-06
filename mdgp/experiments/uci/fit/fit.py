from torch import Tensor


import torch 
from torch.utils.data.dataloader import DataLoader, default_collate
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from gpytorch.metrics import negative_log_predictive_density
from gpytorch.models.deep_gps import DeepGP
from tqdm.autonotebook import tqdm
from math import ceil
from mdgp.experiments.uci.data.datasets import UCIDataset
from mdgp.experiments.uci.fit.metrics import test_log_likelihood, mean_squared_error, negative_log_predictive_density
from dataclasses import dataclass, field


# Settings from the paper 
BATCH_SIZE = 10_000
NUM_ITERATIONS = 20_000
LR = 0.01


@dataclass
class FitArguments: 

    batch_size: int = field(default=BATCH_SIZE, metadata={'help': 'Batch size for training.'})
    num_iterations: int = field(default=NUM_ITERATIONS, metadata={'help': 'Number of iterations to train the model for.'})

    def get_batch_size(self, dataset: UCIDataset) -> int:
        return min(self.batch_size, dataset.train_x.size(0))

    def num_epochs(self, dataset: UCIDataset) -> int:
        iterations_per_epoch = ceil(dataset.train_x.size(0) / self.get_batch_size(dataset))
        return ceil(self.num_iterations / iterations_per_epoch)


def to_device(collate_fn, device: torch.device):
    def to_device_fn(x: Tensor) -> Tensor:
        return tuple(_x.to(device) for _x in collate_fn(x))
    return to_device_fn


def train_step(x: Tensor, y: Tensor, model: DeepGP, optimizer: torch.optim.Optimizer, elbo: VariationalELBO) -> float:
    optimizer.zero_grad()
    output = model(x)
    loss = elbo(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(dataset: UCIDataset, model: DeepGP, fit_args: FitArguments, device: torch.device) -> list[float]: 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, maximize=True)
    elbo = DeepApproximateMLL(VariationalELBO(model.likelihood, model, dataset.train_y.size(0)))
    train_loader = DataLoader(dataset.train_dataset, batch_size=fit_args.get_batch_size(dataset), 
                              shuffle=True, collate_fn=to_device(default_collate, device))

    losses = []
    for _ in (pbar := tqdm(range(fit_args.num_epochs(dataset)), desc='Epochs')):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            loss = train_step(x=x_batch, y=y_batch, model=model, optimizer=optimizer, elbo=elbo)
            epoch_loss += loss
        losses.append(epoch_loss)
        pbar.set_postfix({'ELBO': epoch_loss})

    return losses 


def evaluate(dataset: UCIDataset, model: DeepGP, device: torch.device) -> dict[str, float]:
    with torch.no_grad():
        test_x = dataset.test_x.to(device)
        test_y = dataset.test_y.to(device)
        test_y_std = dataset.test_y_std.to(device)

        out = model.likelihood(model(test_x))
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
