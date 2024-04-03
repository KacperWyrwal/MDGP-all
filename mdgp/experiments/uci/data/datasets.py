"""
UCI Datasets used in the Doubly-Stochastic Variational Inference (DSVI) paper, with processing as described in the paper.
"""
# types
from torch import Tensor


# imports 
import os 
import torch 
import pandas as pd
from torch.utils.data import TensorDataset, Dataset
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


__all__ = ['UCIDataset', 'Kin8mn', 'Power', 'Concrete']


# Settings from the paper 
TEST_SIZE = 0.1


def normalize(x: Tensor) -> Tensor:
    return (x - x.mean(dim=0)) / x.std(dim=0, keepdim=True)


def train_test_split(x: Tensor, y: Tensor, test_size: float = TEST_SIZE) -> tuple[Tensor, Tensor, Tensor, Tensor]: 
    """
    Split the dataset into train and test sets.
    """
    split_idx = int(test_size * x.size(0))
    return x[split_idx:], y[split_idx:], x[:split_idx], y[:split_idx]


def joint_shuffle(*args: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    perm_idx = torch.randperm(args[0].size(0), generator=generator)
    return tuple(x[perm_idx] for x in args)


class UCIDataset:

    UCI_BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

    def __init__(self, name: str, url: str, num_outputs: int = 1, path: str = '../../data/uci/', seed: int | None = None): 
        self.generator = torch.Generator()
        if seed is not None: 
            self.generator.manual_seed(seed)

        self.name = name 
        self.url = url
        self.num_outputs = num_outputs
        self.path = path 
        self.csv_path = os.path.join(self.path, self.name + '.csv')

        # Load, shuffle, split, and normalize data (input and outputs).
        x, y = self.load_data()
        x, y = joint_shuffle(x, y, generator=self.generator)     
        train_x, train_y, test_x, test_y = train_test_split(x, y)
        self.train_x, self.train_y, self.test_x, self.test_y = map(normalize, (train_x, train_y, test_x, test_y))

        # Keeping the stardard deviation allows up to "restore the output scaling for evaluation" (from paper)
        self.test_y_std = test_y.std(dim=0, keepdim=True)
        
    @property
    def dimension(self) -> int:
        return self.train_x.shape[-1]

    @property 
    def train_dataset(self) -> Dataset:
        return TensorDataset(self.train_x, self.train_y)
    
    @property
    def test_dataset(self) -> Dataset:
        return TensorDataset(self.test_x, self.test_y)
    
    def download_data(self) -> None:
        NotImplementedError

    def read_data(self) -> tuple[Tensor, Tensor]:
        xy = torch.from_numpy(pd.read_csv(self.csv_path).values).to(torch.get_default_dtype())
        return xy[:, :-self.num_outputs], xy[:, -self.num_outputs:]

    def load_data(self, overwrite: bool = False) -> tuple[Tensor, Tensor]:
        if overwrite or not os.path.isfile(self.csv_path):
            self.download_data()
        return self.read_data()


class Kin8mn(UCIDataset):

    DEFAULT_URL = 'https://raw.githubusercontent.com/liusiyan/UQnet/master/datasets/UCI_datasets/kin8nm/dataset_2175_kin8nm.csv'

    def __init__(self, path: str = '../../data/uci/', seed: int | None = None, url: str | None = None):
        url = url or Kin8mn.DEFAULT_URL
        super().__init__(name='kin8nm', path=path, url=url, num_outputs=1, seed=seed)

    def download_data(self) -> None:
        df = pd.read_csv(self.url)
        os.makedirs(self.path, exist_ok=True)
        df.to_csv(self.csv_path, index=False)


class Power(UCIDataset):

    DEFAULT_URL = UCIDataset.UCI_BASE_URL + "00294/CCPP.zip"

    def __init__(self, path: str = '../../data/uci/', seed: int | None = None, url: str | None = None):
        url = url or Power.DEFAULT_URL
        super().__init__(name='power', path=path, url=url, num_outputs=1, seed=seed)

    def download_data(self):
        with urlopen(self.url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        df = pd.read_excel('/tmp/CCPP//Folds5x2_pp.xlsx')
        os.makedirs(self.path, exist_ok=True)
        df.to_csv(self.csv_path, index=False)


class Concrete(UCIDataset):

    DEFAULT_URL = UCIDataset.UCI_BASE_URL + 'concrete/compressive/Concrete_Data.xls'

    def __init__(self, path: str = '../../data/uci/', seed: int | None = None, url: str | None = None):
        url = url or Concrete.DEFAULT_URL
        super().__init__(name='concrete', path=path, url=url, num_outputs=1, seed=seed)

    def download_data(self):
        df = pd.read_excel(self.url)
        os.makedirs(self.path, exist_ok=True)
        df.to_csv(self.csv_path, index=False)


class Energy(UCIDataset):
    DEFAULT_URL = UCIDataset.UCI_BASE_URL + '00242/ENB2012_data.xlsx'

    def __init__(self, path: str = '../../data/uci/', seed: int | None = None, url: str | None = None):
        url = url or Energy.DEFAULT_URL
        super().__init__(name='energy', path=path, url=url, num_outputs=1, seed=seed)

    def download_data(self):
        df = pd.read_excel(self.url).drop(columns='Y2')
        os.makedirs(self.path, exist_ok=True)
        df.to_csv(self.csv_path, index=False)


class EnergyDSVI(UCIDataset):
    DEFAULT_URL = UCIDataset.UCI_BASE_URL + '00242/ENB2012_data.xlsx'

    def __init__(self, path: str = '../../data/uci/', seed: int | None = None, url: str | None = None):
        url = url or Energy.DEFAULT_URL
        super().__init__(name='energy', path=path, url=url, num_outputs=1, seed=seed)

    def download_data(self):
        df = pd.read_excel(self.url).drop(columns='Y2')
        os.makedirs(self.path, exist_ok=True)
        df.to_csv(self.csv_path, index=False)


class Yacht(UCIDataset):
    DEFAULT_URL = UCIDataset.UCI_BASE_URL + '00243/yacht_hydrodynamics.data'

    def __init__(self, path: str = '../../data/uci/', seed: int | None = None, url: str | None = None):
        url = url or Yacht.DEFAULT_URL
        super().__init__(name='yacht', path=path, url=url, num_outputs=1, seed=seed)

    def download_data(self):
        df = pd.read_csv(self.url, delim_whitespace=True, header=None)
        os.makedirs(self.path, exist_ok=True)
        df.to_csv(self.csv_path, index=False)


class Boston(UCIDataset):

    DEFAULT_URL = UCIDataset.UCI_BASE_URL + 'housing/housing.data'

    def __init__(self, path: str = '../../data/uci/', seed: int | None = None, url: str | None = None):
        url = url or Boston.DEFAULT_URL
        super().__init__(name='boston', path=path, url=url, num_outputs=1, seed=seed)

    def download_data(self):
        df = pd.read_csv(self.url, delim_whitespace=True, header=None)
        os.makedirs(self.path, exist_ok=True)
        df.to_csv(self.csv_path, index=False)