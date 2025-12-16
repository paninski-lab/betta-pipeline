from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def parse_splits(splits_file: Path) -> dict[str, list[str]]:
    """Parse a simple YAML splits file with keys: train/test/val."""
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")

    # Some splits.yaml in this repo are YAML-ish but also include comments.
    # Use a forgiving parser by stripping inline comments.
    current = None
    splits: dict[str, list[str]] = {"train": [], "test": [], "val": []}
    for raw_line in splits_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            key = line[:-1].strip()
            current = key if key in splits else None
            continue
        if current and line.startswith("-"):
            value = line[1:].strip()
            if value:
                splits[current].append(value)
    return splits


def resolve_session_alias(session: str, subtype: str) -> str:
    """Resolve session alias to on-disk filename conventions."""
    if subtype == "ornamental":
        # e.g. splits.yaml sometimes uses 2.7.1oR / 3.5.1oR, but files are 2.7.1R / 3.5.1R
        if session.endswith("oR"):
            return session[:-2] + "R"
    return session


_HP_RE = re.compile(r"^nh(?P<nh>\d+)_ly(?P<ly>\d+)_lg(?P<lg>\d+)_lr(?P<lr>[\deE\-\+\.]+)$")


def parse_hparam_key(hparam_key: str) -> dict[str, Any]:
    """Parse keys like `nh64_ly3_lg8_lr5e-04`."""
    m = _HP_RE.match(hparam_key)
    if not m:
        raise ValueError(f"Unrecognized hparam_key format: {hparam_key}")
    d = m.groupdict()
    return {
        "num_hid_units": int(d["nh"]),
        "num_layers": int(d["ly"]),
        "num_lags": int(d["lg"]),
        "lr": float(d["lr"]),
    }


def build_binary_dtcn_config(
    *,
    data_path: Path,
    expt_ids: list[str],
    input_size: int,
    output_size: int,
    num_hid_units: int,
    num_layers: int,
    num_lags: int,
    lr: float,
    num_epochs: int,
    seed: int = 43,
    input_dir: str = "features",
    batch_size: int = 16,
    num_workers: int = 4,
    sequence_length: int = 1000,
    train_probability: float = 0.95,
    val_probability: float = 0.05,
) -> dict[str, Any]:
    """Build a lightning-action config dict (binary flare vs background)."""
    return {
        "data": {
            "data_path": str(data_path),
            "input_dir": input_dir,
            "transforms": ["ZScore"],
            "expt_ids": expt_ids,
            "seed": seed,
            "ignore_index": -100,
            "weight_classes": True,
            "label_names": ["background", "flare"],
        },
        "model": {
            "backbone": "dtcn",
            "input_size": input_size,
            "output_size": output_size,
            "num_hid_units": num_hid_units,
            "num_layers": num_layers,
            "num_lags": num_lags,
            "seed": seed,
            # NOTE: do NOT set training.sequence_pad; Model.from_config computes model.sequence_pad.
        },
        "optimizer": {
            "type": "Adam",
            "lr": lr,
            "wd": 0,
            "scheduler": None,
        },
        "training": {
            "device": "gpu",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "sequence_length": sequence_length,
            "train_probability": train_probability,
            "val_probability": val_probability,
        },
    }


def save_config_yaml(config: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


