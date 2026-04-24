"""
Dataset download, organization, preprocessing, and split generation for VisionExtract.

This script downloads multiple KaggleHub datasets, organizes them under domain
folders, converts annotations into YOLO detection labels and binary masks, and
splits them into train/val/test sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.domain_specs import (
    DATASETS,
    DOMAINS,
    datasets_root,
    domain_mask_dataset_root,
    domain_yolo_dataset_root,
    yolo_config_path,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_HINTS = ("mask", "masks", "seg", "segment", "annotation", "annotations", "label")
LABEL_HINTS = ("label", "labels", "annotation", "annotations")


def ensure_structure() -> None:
    for domain in DOMAINS:
        raw_root = datasets_root() / domain / "raw"
        for split in ("train", "val", "test"):
            for branch in ("images", "labels"):
                (domain_yolo_dataset_root(domain) / branch / split).mkdir(parents=True, exist_ok=True)
            for branch in ("images", "masks"):
                (domain_mask_dataset_root(domain) / branch / split).mkdir(parents=True, exist_ok=True)
        raw_root.mkdir(parents=True, exist_ok=True)


def try_import_kagglehub():
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for dataset downloads. Install it with "
            "`./venv/bin/pip install kagglehub`."
        ) from exc
    return kagglehub


def download_datasets(force: bool = False) -> None:
    kagglehub = try_import_kagglehub()
    ensure_structure()

    for spec in DATASETS:
        target_dir = datasets_root() / spec.domain / "raw" / spec.name
        if target_dir.exists() and any(target_dir.iterdir()) and not force:
            print(f"[skip] {spec.name}: raw dataset already present at {target_dir}")
            continue

        source_path = Path(kagglehub.dataset_download(spec.kaggle_id))
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_path, target_dir)
        print(f"[ok] downloaded {spec.kaggle_id} -> {target_dir}")


def normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", "_label", "-label", "_annotation", "-annotation"):
        stem = stem.replace(token, "")
    return stem


def is_mask_path(path: Path) -> bool:
    parts = "/".join(part.lower() for part in path.parts)
    return any(hint in parts for hint in MASK_HINTS)


def is_label_text(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    parts = "/".join(part.lower() for part in path.parts)
    return any(hint in parts for hint in LABEL_HINTS)


def collect_records(raw_dataset_dir: Path) -> List[Dict[str, Optional[Path]]]:
    images: Dict[str, Path] = {}
    masks: Dict[str, Path] = {}
    labels: Dict[str, Path] = {}

    for path in raw_dataset_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        key = normalized_stem(path)

        if suffix in IMAGE_SUFFIXES:
            if is_mask_path(path):
                masks[key] = path
            else:
                images.setdefault(key, path)
        elif is_label_text(path):
            labels[key] = path

    records = []
    for key, image_path in images.items():
        records.append(
            {
                "key": key,
                "image": image_path,
                "mask": masks.get(key),
                "label": labels.get(key),
            }
        )
    return records


def split_for_key(key: str) -> str:
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def resize_image_and_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    image_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    resized_image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if mask is None:
        return resized_image, None
    resized_mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def load_mask(mask_path: Optional[Path], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask_path is None:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != image_shape:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def yolo_from_mask(mask: np.ndarray) -> List[str]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    labels = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {bw / w:.6f} {bh / h:.6f}")
    return labels


def normalize_yolo_label_file(label_path: Path) -> List[str]:
    lines = []
    for raw_line in label_path.read_text().splitlines():
        parts = raw_line.split()
        if len(parts) >= 5:
            lines.append(" ".join(parts[:5]))
    return lines


def save_record(
    domain: str,
    dataset_name: str,
    record: Dict[str, Optional[Path]],
    image_size: int,
) -> Optional[Dict[str, str]]:
    image_path = record["image"]
    if image_path is None:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    mask = load_mask(record["mask"], image.shape[:2])
    resized_image, resized_mask = resize_image_and_mask(image, mask, image_size)
    split = split_for_key(f"{dataset_name}:{record['key']}")
    filename = f"{dataset_name}_{record['key']}.png"
    label_filename = f"{dataset_name}_{record['key']}.txt"

    yolo_image_path = domain_yolo_dataset_root(domain) / "images" / split / filename
    yolo_label_path = domain_yolo_dataset_root(domain) / "labels" / split / label_filename
    mask_image_path = domain_mask_dataset_root(domain) / "images" / split / filename
    mask_label_path = domain_mask_dataset_root(domain) / "masks" / split / filename

    cv2.imwrite(str(yolo_image_path), resized_image)
    cv2.imwrite(str(mask_image_path), resized_image)

    if resized_mask is not None:
        cv2.imwrite(str(mask_label_path), resized_mask)
        yolo_labels = yolo_from_mask(resized_mask)
    elif record["label"] is not None:
        yolo_labels = normalize_yolo_label_file(record["label"])
    else:
        yolo_labels = []

    yolo_label_path.write_text("\n".join(yolo_labels))
    if resized_mask is None:
        cv2.imwrite(str(mask_label_path), np.zeros((image_size, image_size), dtype=np.uint8))

    return {
        "domain": domain,
        "dataset": dataset_name,
        "split": split,
        "image": str(yolo_image_path),
        "label": str(yolo_label_path),
        "mask": str(mask_label_path),
    }


def process_datasets(image_size: int) -> None:
    ensure_structure()
    manifest: List[Dict[str, str]] = []

    for spec in DATASETS:
        raw_dir = datasets_root() / spec.domain / "raw" / spec.name
        if not raw_dir.exists():
            print(f"[warn] raw dataset missing for {spec.name}: {raw_dir}")
            continue

        records = collect_records(raw_dir)
        print(f"[info] {spec.name}: discovered {len(records)} candidate samples")
        for record in records:
            saved = save_record(spec.domain, spec.name, record, image_size)
            if saved:
                manifest.append(saved)

    (datasets_root() / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_domain_yamls()
    print(f"[ok] processed {len(manifest)} samples across all domains")


def write_domain_yamls() -> None:
    names_map = {
        "general": {0: "object"},
        "bio": {0: "cell"},
        "space": {0: "feature"},
    }
    for domain in DOMAINS:
        payload = {
            "path": str(domain_yolo_dataset_root(domain)),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names_map[domain],
        }
        with yolo_config_path(domain).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess VisionExtract datasets.")
    parser.add_argument("--download", action="store_true", help="Download datasets from KaggleHub.")
    parser.add_argument("--force-download", action="store_true", help="Re-download raw datasets.")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess and split datasets.")
    parser.add_argument("--image-size", type=int, default=640, help="Output image size for preprocessing.")
    args = parser.parse_args()

    if args.download:
        download_datasets(force=args.force_download)
    if args.preprocess:
        process_datasets(image_size=args.image_size)
    if not args.download and not args.preprocess:
        parser.error("Select at least one action: --download and/or --preprocess")


if __name__ == "__main__":
    main()
