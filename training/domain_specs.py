"""Shared domain dataset metadata for download and training scripts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    name: str
    kaggle_id: str


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        domain="bio",
        name="bccd_with_mask",
        kaggle_id="jeetblahiri/bccd-dataset-with-mask",
    ),
    DatasetSpec(
        domain="space",
        name="spacenet_si",
        kaggle_id="sabermalek/spacenetsi",
    ),
    DatasetSpec(
        domain="general",
        name="image_segmentation_dataset",
        kaggle_id="mnavaidd/image-segmentation-dataset",
    ),
]

DOMAINS = ("general", "bio", "space")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def datasets_root() -> Path:
    return repo_root() / "datasets"


def models_root() -> Path:
    return repo_root() / "models"


def configs_root() -> Path:
    return repo_root() / "training" / "configs"


def yolo_config_path(domain: str) -> Path:
    return configs_root() / f"{domain}_yolo.yaml"


def domain_processed_root(domain: str) -> Path:
    return datasets_root() / domain / "processed"


def domain_yolo_dataset_root(domain: str) -> Path:
    return domain_processed_root(domain) / "yolo"


def domain_mask_dataset_root(domain: str) -> Path:
    return domain_processed_root(domain) / "masks"


def model_output_root(domain: str, model_type: str) -> Path:
    return models_root() / domain / model_type
