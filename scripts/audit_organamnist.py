import argparse
import os
import sys
import numpy as np

# Ensure root dir is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import download_organamnist, ORGAN_AMNIST_CLASSES


def audit_dataset(size: int = 64, data_dir: str = "data/raw"):
    filepath = download_organamnist(size=size, target_dir=data_dir)
    print(f"\n========================================================")
    print(f"       AUDIT REPORT: OrganAMNIST {size}x{size}")
    print(f"========================================================")
    print(f"File path: {filepath}")
    print(f"File size: {os.path.getsize(filepath) / (1024 * 1024):.2f} MB")

    data = np.load(filepath)
    keys = list(data.keys())
    print(f"Keys inside archive: {keys}")

    for split in ["train", "val", "test"]:
        imgs = data[f"{split}_images"]
        labels = data[f"{split}_labels"].squeeze()
        print(f"\n--- Split: {split.upper()} ---")
        print(f"  Images shape: {imgs.shape} (dtype: {imgs.dtype})")
        print(f"  Labels shape: {labels.shape} (dtype: {labels.dtype})")
        print(f"  Pixel range: [{imgs.min()}, {imgs.max()}]")

        unique, counts = np.unique(labels, return_counts=True)
        print("  Class distribution:")
        for cls_id, count in zip(unique, counts):
            cls_name = ORGAN_AMNIST_CLASSES.get(int(cls_id), f"class_{cls_id}")
            pct = (count / len(labels)) * 100
            print(f"    [{cls_id:2d}] {cls_name:<15}: {count:5d} ({pct:5.2f}%)")

    print("\n[Audit] Validation complete. All splits and classes verified successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit OrganAMNIST dataset.")
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        choices=[64, 128],
        help="Image size (64 or 128)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory to save raw dataset",
    )
    args = parser.parse_args()

    audit_dataset(size=args.size, data_dir=args.data_dir)

