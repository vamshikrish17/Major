"""Evaluate trained VisionExtract models and save a consolidated summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.domain_specs import DOMAINS, model_output_root, models_root


def load_json_if_exists(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def main() -> None:
    summary = {"domains": {}}

    for domain in DOMAINS:
        domain_summary = {}

        yolo_metrics = load_json_if_exists(model_output_root(domain, "yolo") / "metrics.json")
        if yolo_metrics is not None:
            domain_summary["yolo"] = yolo_metrics

        if domain == "bio":
            unet_metrics = load_json_if_exists(model_output_root(domain, "unet") / "metrics.json")
            if unet_metrics is not None:
                domain_summary["unet"] = unet_metrics

        summary["domains"][domain] = domain_summary

    output_path = models_root() / "evaluation_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"[ok] wrote evaluation summary -> {output_path}")


if __name__ == "__main__":
    main()
