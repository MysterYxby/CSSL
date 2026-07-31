from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy import io
from torch.utils.data import Dataset

from network.wald_utilities import genMTF_MS, genMTF_PAN


class RSDataset(Dataset):
    def __init__(self, data_dir, resolution="RR", satellite="worldview-2"):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.satellite = satellite
        self.max_val = 1023 if "gaofen" in str(self.data_dir).lower() else 2047

        self.ms_dir = self.data_dir / "MS"
        self.pan_dir = self.data_dir / "PAN"
        self.label_dir = self.data_dir / "label"
        for directory, name in [(self.ms_dir, "MS"), (self.pan_dir, "PAN")]:
            if not directory.exists():
                raise FileNotFoundError(f"{name} directory not found: {directory}")

        self.mat_files = sorted(file.name for file in self.ms_dir.glob("*.mat"))
        if not self.mat_files:
            raise FileNotFoundError(f"No .mat files found in {self.ms_dir}")

        if resolution == "Hybrid":
            base_names = [file[:-4] for file in self.mat_files]
            self.txt_files = [f"{base_name}.txt" for base_name in base_names]
            missing = [file for file in self.txt_files if not (self.label_dir / file).exists()]
            if missing:
                raise FileNotFoundError(f"Label files missing: {missing}")
        else:
            self.txt_files = None

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        name = self.mat_files[idx]
        ms = io.loadmat(self.ms_dir / name)["MS"]
        pan = io.loadmat(self.pan_dir / name)["PAN"]
        ms = torch.from_numpy(ms.astype(np.float32)).permute(2, 0, 1)
        pan = torch.from_numpy(pan.astype(np.float32)).unsqueeze(0)
        gt = ms.clone()
        bands = ms.shape[0]

        if self.resolution == "FR":
            up_ms = F.interpolate(ms.unsqueeze(0), size=pan.shape[-2:], mode="bicubic").squeeze(0)
            return ms / self.max_val, pan / self.max_val, up_ms / self.max_val

        if self.resolution == "RR":
            ms = genMTF_MS(ms.unsqueeze(0), 4, self.satellite, bands).squeeze(0)
            pan = genMTF_PAN(pan.unsqueeze(0), 4, self.satellite).squeeze(0)
            up_ms = F.interpolate(ms.unsqueeze(0), size=pan.shape[-2:], mode="bicubic").squeeze(0)
            return ms / self.max_val, pan / self.max_val, up_ms / self.max_val, gt / self.max_val

        if self.resolution == "Hybrid":
            up_ms = F.interpolate(ms.unsqueeze(0), size=pan.shape[-2:], mode="bicubic").squeeze(0)
            txt_name = self.txt_files[idx]
            with open(self.label_dir / txt_name, encoding="utf-8") as file:
                label = int(file.read().strip() or -1)
            if label < 0:
                raise ValueError(f"Invalid label in {txt_name}")
            return ms / self.max_val, pan / self.max_val, up_ms / self.max_val, torch.tensor(label, dtype=torch.long)

        raise ValueError(f"Unknown resolution: {self.resolution}")


class PansharpeningDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset=None, batch_size=16, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


def reader_testdata(data_dir, filename, sensor, fr_test=False):
    data_dir = Path(data_dir)
    max_val = 1023 if "gaofen" in str(data_dir).lower() else 2047
    mat_ms = io.loadmat(data_dir / "MS" / filename)
    mat_pan = io.loadmat(data_dir / "PAN" / filename)
    ms = torch.from_numpy((mat_ms["MS"] / max_val).astype(np.float32)).permute(2, 0, 1)
    pan = torch.from_numpy((mat_pan["PAN"] / max_val).astype(np.float32)).unsqueeze(0)
    gt = ms.clone()
    if not fr_test:
        bands = ms.shape[0]
        ms = genMTF_MS(ms.unsqueeze(0), 4, sensor, bands).squeeze(0)
        pan = genMTF_PAN(pan.unsqueeze(0), 4, sensor).squeeze(0)
    up_ms = F.interpolate(ms.unsqueeze(0), size=pan.shape[-2:], mode="bicubic").squeeze(0)
    return ms, up_ms, pan, gt
