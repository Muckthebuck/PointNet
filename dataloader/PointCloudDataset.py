from typing import List, Tuple, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os


class PointCloudDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        phase: str,
        data_dir: str,
        url: Optional[str] = None,
        from_folder: bool = False,
    ):
        self.name = dataset_name.lower()
        self.phase = phase
        self.root = Path(data_dir)
        self.url = url
        self.from_folder = from_folder

        self.dataset_dir = self._resolve_dataset_dir()
        if not self.from_folder:
            self._maybe_download()

        self.data, self.labels, self.extra = self._load()

    def _resolve_dataset_dir(self) -> Path:
        mapping = {
            "shapenet": "shapenet_part_seg_hdf5_data",
            "modelnet": "modelnet40_ply_hdf5_2048",
        }
        if self.name not in mapping:
            raise ValueError(f"Unsupported dataset: {self.name}")
        return self.root / mapping[self.name]

    def _maybe_download(self):
        if self.dataset_dir.exists():
            return
        if not self.url:
            raise ValueError("No URL provided for download.")
        zipfile = Path(self.url).name
        os.system(f"wget --no-check-certificate {self.url}")
        os.system(f"unzip {zipfile}")
        os.system(f"mv {zipfile.replace('.zip', '')} {self.dataset_dir}")
        os.system(f"rm {zipfile}")

    def _load(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        if self.from_folder:
            return self._load_from_folder()
        if self.name == "shapenet":
            return self._load_from_file_list("pid")
        if self.name == "modelnet":
            return self._load_from_file_list("normal", use_val_as_test=True)
        raise ValueError(f"Unsupported dataset: {self.name}")

    def _load_from_file_list(
        self, extra_key: str = "", use_val_as_test: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        phase = "test" if use_val_as_test and self.phase == "val" else self.phase
        file_list = self.dataset_dir / f"{phase}_hdf5_file_list.txt" if self.name == "shapenet" else self.dataset_dir / f"{phase}_files.txt"

        with open(file_list) as f:
            files = [self.dataset_dir / Path(line.strip()).name for line in f]

        data, labels, extras = [], [], []

        for file in files:
            with h5py.File(file, "r") as f:
                data.append(f["data"][:])
                labels.append(f["label"][:])
                if extra_key and extra_key in f:
                    extras.append(f[extra_key][:])

        return (
            np.concatenate(data).astype(np.float32),
            np.concatenate(labels).astype(np.int_),
            np.concatenate(extras).astype(np.int_) if extras else None,
        )

    def _load_from_folder(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        folder = self.dataset_dir / self.phase
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        data, labels, extras = [], [], []
        for file in sorted(folder.glob("*.h5")):
            with h5py.File(file, "r") as f:
                data.append(f["data"][:])
                labels.append(f["label"][:])
                if self.name == "shapenet" and "pid" in f:
                    extras.append(f["pid"][:])

        return (
            np.concatenate(data).astype(np.float32),
            np.concatenate(labels).astype(np.int_),
            np.concatenate(extras).astype(np.int_) if extras else None,
        )

    def _normalise(self, pc: np.ndarray) -> np.ndarray:
        pc -= pc.mean(0)
        pc /= np.max(np.sqrt((pc**2).sum(1)))
        return pc

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        pc = torch.from_numpy(self._normalise(self.data[idx]))
        label = torch.tensor(self.labels[idx])
        if self.name == "shapenet" and self.extra is not None:
            pc_label = torch.tensor(self.extra[idx])
            return pc, pc_label, label
        return pc, label

    def __len__(self) -> int:
        return len(self.data)


def get_data_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    url: Optional[str] = None,
    phases: List[str] = ["train", "val", "test"],
    from_folder: bool = False,
) -> Tuple[List[PointCloudDataset], List[DataLoader]]:
    datasets, loaders = [], []
    for phase in phases:
        ds = PointCloudDataset(
            dataset_name, phase, data_dir, url=url, from_folder=from_folder
        )
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=phase == "train",
            drop_last=phase == "train",
        )
        datasets.append(ds)
        loaders.append(dl)
    return datasets, loaders
