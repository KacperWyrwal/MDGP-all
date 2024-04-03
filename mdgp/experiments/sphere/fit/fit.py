import torch 
import gpytorch 
import geoopt
from torch import Tensor 
from torch.utils.data import DataLoader, default_collate
from gpytorch.models.deep_gps import DeepGP
from gpytorch.mlls import VariationalELBO
from gpytorch.metrics import mean_squared_error, negative_log_predictive_density
from dataclasses import dataclass, field
from tqdm.auto import tqdm 
from mdgp.experiments.sphere.data import SphereDataset
from mdgp.experiments.sphere.fit.metrics import test_log_likelihood, squared_error_batch, test_log_likelihood_batch
from mdgp.experiments.sphere.model.model import ModelArguments


LR = 0.01


@dataclass
class FitArguments: 
    num_epochs: int = field(default=1000, metadata={'help': 'Number of steps to train for'})
    train_batch_size: int = field(default=1024, metadata={'help': 'Batch size for training.'})
    test_batch_size: int | None = field(default=None, metadata={'help': 'Batch size for testing.'})
    train_num_samples: int = field(default=10)
    test_num_samples: int = field(default=10)

    def __post_init__(self):
        if self.test_batch_size is None:
            self.test_batch_size = self.train_batch_size

    def get_test_batch_size(self, dataset: SphereDataset) -> int:
        return min(self.test_batch_size, dataset.test_x.size(0))

    def get_train_batch_size(self, dataset: SphereDataset) -> int:
        return min(self.train_batch_size, dataset.train_x.size(0))


def train_step(x: Tensor, y: Tensor, model: DeepGP, optimizer: torch.optim.Optimizer, elbo: VariationalELBO) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = elbo(output, y).neg()
    loss.backward()
    optimizer.step()
    return loss.item()


def to_device(collate_fn, device: torch.device):
    def to_device_fn(x: Tensor) -> Tensor:
        return tuple(_x.to(device) for _x in collate_fn(x))
    return to_device_fn


def train(dataset: SphereDataset, model: DeepGP, fit_args: FitArguments, device: torch.device, model_args: ModelArguments | None = None) -> list[float]: 
    with gpytorch.settings.num_likelihood_samples(fit_args.train_num_samples):
        if model_args is not None and model_args.need_geometry_aware_optimizer:
            print("Using RiemannianAdam")
            optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=LR)
        else:
            print("Using Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        elbo = gpytorch.mlls.DeepApproximateMLL(VariationalELBO(model.likelihood, model, dataset.train_y.size(0)))
        train_loader = DataLoader(dataset.train_dataset, batch_size=fit_args.get_train_batch_size(dataset), 
                                shuffle=True, collate_fn=to_device(default_collate, device))

        losses = []
        for _ in (pbar := tqdm(range(fit_args.num_epochs), desc='Epochs', position=0, leave=True)):
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                loss = train_step(x=x_batch, y=y_batch, model=model, optimizer=optimizer, elbo=elbo)
                epoch_loss += loss
            losses.append(epoch_loss)
            pbar.set_postfix({'ELBO': epoch_loss})

        return losses 


def evaluate_shallow(dataset: SphereDataset, model: DeepGP, device: torch.device, fit_args: FitArguments | None = None) -> dict[str, float]:
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(fit_args.test_num_samples):
        test_x, test_y = dataset.test_x.to(device), dataset.test_y.to(device)

        out = model.likelihood(model(test_x))
        tll = test_log_likelihood(out, test_y)
        mse = mean_squared_error(out, test_y)
        nlpd = negative_log_predictive_density(out, test_y)
        metrics = {
            'tll': tll.mean().item(), 
            'mse': mse.mean().item(),
            'nlpd': nlpd.mean().item(),
        }
        print(f"TLL: {metrics['tll']}, MSE: {metrics['mse']}, NLPD: {metrics['nlpd']}")
    return metrics 


def evaluate_deep(dataset: SphereDataset, model: DeepGP, device: torch.device, fit_args: FitArguments) -> dict[str, float]:
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(fit_args.test_num_samples):
        test_loader = DataLoader(dataset.test_dataset, batch_size=fit_args.get_test_batch_size(dataset), 
                                 shuffle=True, collate_fn=to_device(default_collate, device))
        tll, se = [], []
        for x_batch, y_batch in test_loader:
            out = model.likelihood(model(x_batch))
            tll.append(test_log_likelihood_batch(out, y_batch))
            se.append(squared_error_batch(out, y_batch))
        tll = torch.cat(tll, dim=-1).mean().item()
        mse = torch.cat(se, dim=-1).mean().item()
        print(f"TLL: {tll}, MSE: {mse}")
    return {'tll': tll, 'mse': mse}
    