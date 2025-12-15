from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightning_action.api import Model


def find_repo_root(start: Path | None = None) -> Path:
    """Find repository root by locating 'Week 14/hyperparameter_sweep'."""
    candidates = []
    if start is not None:
        candidates.append(start)
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve())

    for base in candidates:
        base = base.resolve()
        for p in [base] + list(base.parents):
            if (p / "Week 14" / "hyperparameter_sweep").exists():
                return p
    # fallback: current working directory
    return Path.cwd().resolve()


def parse_splits(splits_file: Path) -> dict[str, list[str]]:
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
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
    if subtype == "ornamental" and session.endswith("oR"):
        return session[:-2] + "R"
    return session


def best_hparam_key_from_results(results_file: Path, *, nested_subtype: str | None = None) -> str:
    data = json.loads(results_file.read_text(encoding="utf-8"))
    if nested_subtype is None:
        d = data.get("hparam_results", {})
    else:
        d = data.get("subtype_results", {}).get(nested_subtype, {})
    if not d:
        raise RuntimeError(f"No results found in {results_file} (nested_subtype={nested_subtype})")
    return max(d.keys(), key=lambda k: d[k].get("test_f1", float("-inf")))


def global_best_aggregate_key(results_file: Path, subtypes: list[str]) -> str:
    data = json.loads(results_file.read_text(encoding="utf-8"))
    subtype_results = data.get("subtype_results", {})
    keys = None
    for st in subtypes:
        st_keys = set(subtype_results.get(st, {}).keys())
        keys = st_keys if keys is None else (keys & st_keys)
    if not keys:
        raise RuntimeError("No common aggregate hparam keys across all fish types.")

    def mean_f1(k: str) -> float:
        return sum(subtype_results[st][k]["test_f1"] for st in subtypes) / len(subtypes)

    return max(keys, key=mean_f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lightning-Action inference (Week 14 hyperparameter_sweep layout).")
    parser.add_argument(
        "--project_root",
        type=Path,
        default=None,
        help="Repo root (auto-detected if omitted). Should contain 'Week 14/hyperparameter_sweep/'.",
    )
    parser.add_argument("--variant", type=str, default="LP_with_cal_contour")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help=(
            "Override dataset path (must contain features/). "
            "If not provided, defaults to Week 14/hyperparameter_sweep/binary_data/<target_subtype>/<variant>."
        ),
    )
    parser.add_argument("--input_dir", type=str, default="features", help="Input directory name under data_path.")

    parser.add_argument(
        "--model_kind",
        type=str,
        required=True,
        choices=["subtype", "aggregate_single"],
        help="Which model family to use.",
    )
    parser.add_argument(
        "--target_subtype",
        type=str,
        required=True,
        choices=["fighting", "hybrid", "ornamental", "wild"],
        help="Which fish type to run inference on (also selects the dataset).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "val", "all"],
        help="Which split from that subtype's splits.yaml to run inference on.",
    )
    parser.add_argument(
        "--sessions_file",
        type=Path,
        default=None,
        help="Optional splits.yaml to pick sessions from (overrides default dataset splits.yaml).",
    )
    parser.add_argument(
        "--session_ids",
        nargs="+",
        default=None,
        help="Optional explicit list of session IDs (overrides --split/--sessions_file).",
    )
    parser.add_argument(
        "--hparam_key",
        type=str,
        default=None,
        help="Explicit model key like nh64_ly3_lg8_lr5e-04. If omitted, best is selected from results.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to save predictions (defaults under model_dir/predictions/...).",
    )

    args = parser.parse_args()

    project_root = (args.project_root or find_repo_root()).resolve()
    week14_root = project_root / "Week 14"
    sweep_root = week14_root / "hyperparameter_sweep"
    results_root = sweep_root / "results"
    models_root = sweep_root / "models"
    data_root = sweep_root / "binary_data"

    # dataset path (we run inference on the target subtype's dataset by default)
    dataset_path = args.data_path or (data_root / args.target_subtype / args.variant)

    # session selection priority:
    # 1) explicit --session_ids
    # 2) --sessions_file + --split
    # 3) default dataset splits.yaml + --split
    if args.session_ids:
        sessions = args.session_ids
    else:
        splits_file = args.sessions_file or (dataset_path / "splits.yaml")
        splits = parse_splits(splits_file)
        if args.split == "all":
            sessions = splits["train"] + splits["test"] + splits.get("val", [])
        else:
            sessions = splits[args.split]

    # resolve known aliases for on-disk filenames
    sessions = [resolve_session_alias(s, args.target_subtype) for s in sessions]

    # choose model directory
    if args.model_kind == "subtype":
        if args.hparam_key is None:
            args.hparam_key = best_hparam_key_from_results(results_root / f"{args.target_subtype}_results.json")
        model_dir = models_root / args.target_subtype / args.hparam_key
        default_out = model_dir / "predictions"
    else:
        if args.hparam_key is None:
            args.hparam_key = global_best_aggregate_key(
                results_root / "aggregate_results.json",
                ["fighting", "hybrid", "ornamental", "wild"],
            )
        model_dir = models_root / "aggregate" / args.hparam_key
        default_out = model_dir / "predictions" / args.target_subtype

    out_dir = args.output_dir or default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LIGHTNING-ACTION INFERENCE")
    print(f"Model kind: {args.model_kind}")
    print(f"Target subtype: {args.target_subtype}")
    print(f"Variant: {args.variant}")
    print(f"Model key: {args.hparam_key}")
    print(f"Model dir: {model_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Split: {args.split} ({len(sessions)} sessions)")
    print(f"Output: {out_dir}")
    print("=" * 80)

    model = Model.from_dir(model_dir)
    model.predict(
        data_path=str(dataset_path),
        input_dir=args.input_dir,
        output_dir=str(out_dir),
        expt_ids=sessions,
    )


if __name__ == "__main__":
    main()


