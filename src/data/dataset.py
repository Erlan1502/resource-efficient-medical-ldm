import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ORGAN_AMNIST_CLASSES = {
    0: "bladder",
    1: "femur-left",
    2: "femur-right",
    3: "heart",
    4: "kidney-left",
    5: "kidney-right",
    6: "liver",
    7: "lung-left",
    8: "lung-right",
    9: "pancreas",
    10: "spleen",
}

ZENODO_URLS = {
    64: "https://zenodo.org/records/10519652/files/organamnist_64.npz?download=1",
    128: "https://zenodo.org/records/10519652/files/organamnist_128.npz?download=1",
}


def download_organamnist(size: int = 64, target_dir: str = "data/raw") -> str:
    """Download OrganAMNIST npz file from Zenodo if not already present."""
    os.makedirs(target_dir, exist_ok=True)
    filename = f"organamnist_{size}.npz"
    filepath = os.path.join(target_dir, filename)

    if os.path.exists(filepath):
        print(f"[Dataset] File already exists: {filepath}")
        return filepath

    url = ZENODO_URLS.get(size)
    if not url:
        raise ValueError(f"Unsupported size: {size}. Supported: 64, 128")

    print(f"[Dataset] Downloading {filename} from Zenodo...")
    urllib.request.urlretrieve(url, filepath)
    print(f"[Dataset] Download completed: {filepath}")
    return filepath


class OrganAMNISTDataset(Dataset):
    """PyTorch Dataset for OrganAMNIST 2D axial CT slices.

    Normalizes grayscale images from uint8 [0, 255] to float32 [-1, 1].
    Outputs shape: [1, H, W]
    """

    def __init__(
        self,
        split: str = "train",
        size: int = 64,
        data_dir: str = "data/raw",
        auto_download: bool = True,
    ):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        filepath = os.path.join(data_dir, f"organamnist_{size}.npz")
        if not os.path.exists(filepath):
            if auto_download:
                download_organamnist(size=size, target_dir=data_dir)
            else:
                raise FileNotFoundError(f"Dataset file not found at {filepath}")

        data = np.load(filepath)
        self.images = data[f"{split}_images"]  # shape: (N, H, W)
        self.labels = data[f"{split}_labels"].squeeze().astype(np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]  # (H, W), uint8
        # Convert to float32 tensor normalized to [-1, 1]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 127.5 - 1.0
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor


def get_dataloaders(
    size: int = 64,
    batch_size: int = 64,
    data_dir: str = "data/raw",
    num_workers: int = 0,
):
    """Factory to get train, val, and test DataLoaders."""
    train_ds = OrganAMNISTDataset(split="train", size=size, data_dir=data_dir)
    val_ds = OrganAMNISTDataset(split="val", size=size, data_dir=data_dir)
    test_ds = OrganAMNISTDataset(split="test", size=size, data_dir=data_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

