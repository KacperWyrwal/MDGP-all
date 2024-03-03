from dataclasses import dataclass, field 
from mdgp.experiments.uci.data.datasets import *


@dataclass
class DataArguments: 
    dataset_name: str = field(default='kin8nm', metadata={'help': 'Name of the dataset. Must be one of ["kin8nm", "power", "concrete"]'})

    @property
    def dataset(self) -> UCIDataset:
        if self.dataset_name == 'kin8nm':
            return Kin8mn()
        elif self.dataset_name == 'power':
            return Power()
        elif self.dataset_name == 'concrete':
            return Concrete()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
