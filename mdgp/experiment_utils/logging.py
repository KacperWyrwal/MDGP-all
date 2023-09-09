import os 
from lightning.fabric.loggers import CSVLogger as FabricCSVLogger
import warnings 


__all__ = [
    "CSVLogger",
    "log",
    "finalize",
]


class CSVLogger(FabricCSVLogger):
    def _get_next_version(self) -> int:
        root_dir = os.path.join(self.root_dir, self.name)

        if not self._fs.isdir(root_dir):
            warnings.warn(f"Missing logger folder: {root_dir}")
            return 0

        existing_versions = []
        for d in self._fs.listdir(root_dir):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if self._fs.isdir(full_path) and name.startswith("version_"):
                existing_versions.append(int(name.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


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