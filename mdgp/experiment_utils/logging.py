import os 
from lightning.fabric.loggers import CSVLogger as FabricCSVLogger
import warnings 


__all__ = [
    "CSVLogger",
    "log",
    "finalize",
]


class CSVLogger(FabricCSVLogger):

    @property 
    def log_dir(self):
        return self.root_dir


def log(loggers, metrics, step=None):
    if loggers is None:
        return 
    for logger in loggers: 
        logger.log_metrics(metrics=metrics, step=step)


def finalize(loggers): 
    if loggers is None: 
        return 
    for logger in loggers: 
        logger.finalize("Done")