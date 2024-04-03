from dataclasses import dataclass, field 
from mdgp.experiments.uci.data.datasets import Energy, Power, Kin8mn, Concrete, Yacht, UCIDataset


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
        elif self.dataset_name == 'yacht':
            return Yacht()
        elif self.dataset_name == 'energy':
            return Energy()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
