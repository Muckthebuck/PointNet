from typing import List, Tuple, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import subprocess


class PointCloudDataset(Dataset):
    """
    Generic Dataset class for ShapeNet and ModelNet point cloud datasets.

    Args:
        dataset_name (str): One of ['shapenet', 'modelnet']
        phase (str): One of ['train', 'val', 'test']
        data_dir (str): Path to root data directory
        url (str, optional): URL for automatic download if dataset is missing
        from_folder (bool): If True, loads directly from HDF5 folder (for custom datasets)
    """
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
            raise ValueError("No URL provided for dataset download.")
        zipfile = Path(self.url).name
        subprocess.run(["wget", "--no-check-certificate", self.url])
        subprocess.run(["unzip", zipfile])
        extracted_folder = zipfile.replace(".zip", "")
        subprocess.run(["mv", extracted_folder, str(self.dataset_dir)])
        Path(zipfile).unlink(missing_ok=True)

    def _load(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.from_folder:
            return self._load_from_folder()

        if self.name == "shapenet":
            return self._load_from_file_list(extra_key="pid")
        elif self.name == "modelnet":
            return self._load_from_file_list(extra_key="normal", use_val_as_test=True)

        raise ValueError(f"Unsupported dataset: {self.name}")

    def _load_from_file_list(
        self, extra_key: str = "", use_val_as_test: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        phase = "test" if use_val_as_test and self.phase == "val" else self.phase
        list_file = (
            self.dataset_dir / f"{phase}_hdf5_file_list.txt"
            if self.name == "shapenet"
            else self.dataset_dir / f"{phase}_files.txt"
        )

        if not list_file.exists():
            raise FileNotFoundError(f"Missing file list: {list_file}")

        with open(list_file) as f:
            files = [line.strip() for line in f if line.strip()]

        data, labels, extras = [], [], []
        for path in files:
            path = self.dataset_dir / Path(path).name
            with h5py.File(path, "r") as f:
                data.append(f["data"][:])
                labels.append(f["label"][:])
                if extra_key and extra_key in f:
                    extras.append(f[extra_key][:])

        return (
            np.concatenate(data).astype(np.float32),
            np.concatenate(labels).astype(np.int64),
            np.concatenate(extras).astype(np.int64) if extras else None,
        )

    def _load_from_folder(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        folder = self.dataset_dir / self.phase
        if not folder.exists():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")

        data, labels, extras = [], [], []
        for file in sorted(folder.glob("*.h5")):
            with h5py.File(file, "r") as f:
                data.append(f["data"][:])
                labels.append(f["label"][:])
                if self.name == "shapenet" and "pid" in f:
                    extras.append(f["pid"][:])

        return (
            np.concatenate(data).astype(np.float32),
            np.concatenate(labels).astype(np.int64),
            np.concatenate(extras).astype(np.int64) if extras else None,
        )

    def _normalize(self, pc: np.ndarray) -> np.ndarray:
        pc -= pc.mean(0)
        scale = np.max(np.sqrt((pc ** 2).sum(axis=1)))
        return pc / scale

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        pc = self._normalize(self.data[idx])
        pc_tensor = torch.from_numpy(pc)  # (N, 3)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.name == "shapenet" and self.extra is not None:
            seg_label = torch.tensor(self.extra[idx], dtype=torch.long)
            return pc_tensor, seg_label, label

        return pc_tensor, label

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
    """
    Returns dataset objects and their corresponding PyTorch DataLoaders for different phases.
    """
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
