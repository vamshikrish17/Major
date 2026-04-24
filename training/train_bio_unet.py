"""Train a U-Net model on the biological segmentation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.segmentation import LightweightUNet
from training.domain_specs import domain_mask_dataset_root, model_output_root


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SegmentationPairDataset(Dataset):
    def __init__(self, split: str):
        root = domain_mask_dataset_root("bio")
        self.image_dir = root / "images" / split
        self.mask_dir = root / "masks" / split
        self.samples = sorted([p for p in self.image_dir.glob("*.png") if (self.mask_dir / p.name).exists()])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        mask_path = self.mask_dir / image_path.name

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_tensor = (torch.from_numpy(mask).float().unsqueeze(0) / 255.0).clamp(0, 1)
        return image_tensor, mask_tensor


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def segmentation_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum().item()

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "iou": iou,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    train: bool,
) -> Tuple[float, Dict[str, float]]:
    model.train(mode=train)
    bce = nn.BCELoss()
    total_loss = 0.0
    metric_sums = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "iou": 0.0}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            preds = model(images)
            loss = bce(preds, masks) + dice_loss(preds, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        batch_metrics = segmentation_metrics(preds.detach(), masks)
        for key, value in batch_metrics.items():
            metric_sums[key] += value

    steps = max(len(loader), 1)
    avg_metrics = {key: value / steps for key, value in metric_sums.items()}
    return total_loss / steps, avg_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the biological U-Net segmentation model.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=auto_device())
    args = parser.parse_args()

    train_ds = SegmentationPairDataset("train")
    val_ds = SegmentationPairDataset("val")
    test_ds = SegmentationPairDataset("test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LightweightUNet().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_root = model_output_root("bio", "unet")
    output_root.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, args.device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer, args.device, train=False)

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(summary)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), output_root / "best.pt")

    test_loss, test_metrics = run_epoch(model, test_loader, optimizer, args.device, train=False)
    payload = {
        "history": history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "checkpoint": str(output_root / "best.pt"),
    }
    (output_root / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"[ok] saved bio U-Net checkpoint -> {output_root / 'best.pt'}")


if __name__ == "__main__":
    main()
