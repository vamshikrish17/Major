"""Train separate YOLOv8 models for each VisionExtract domain."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.domain_specs import DOMAINS, model_output_root, yolo_config_path


def auto_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_domain(domain: str, model_name: str, epochs: int, imgsz: int, batch: int, device: str) -> None:
    cfg = yolo_config_path(domain)
    if not cfg.exists():
        raise FileNotFoundError(f"Missing YOLO config for {domain}: {cfg}")

    output_root = model_output_root(domain, "yolo")
    output_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    result = model.train(
        data=str(cfg),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_root),
        name="runs",
        exist_ok=True,
        pretrained=True,
        amp=(device != "cpu"),
        cache=True,
    )

    best_weight = Path(result.save_dir) / "weights" / "best.pt"
    if best_weight.exists():
        shutil.copy2(best_weight, output_root / "best.pt")

    metrics = {
        "domain": domain,
        "config": str(cfg),
        "weights": str(output_root / "best.pt"),
        "results": dict(result.results_dict),
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[ok] trained {domain} YOLO model -> {output_root / 'best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 per VisionExtract domain.")
    parser.add_argument("--domain", choices=[*DOMAINS, "all"], default="all")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=auto_device())
    args = parser.parse_args()

    domains = DOMAINS if args.domain == "all" else (args.domain,)
    for domain in domains:
        train_domain(domain, args.model, args.epochs, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
