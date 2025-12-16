from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a standalone script (no package install needed)
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from lightning_action.api import Model

from common import (
    build_binary_dtcn_config,
    parse_hparam_key,
    parse_splits,
    resolve_session_alias,
    save_config_yaml,
)


def load_best_key(results_file: Path) -> str:
    data = json.loads(results_file.read_text(encoding="utf-8"))
    hparam_results = data.get("hparam_results", {})
    if not hparam_results:
        raise RuntimeError(f"No hparam_results found in {results_file}")
    return max(hparam_results.keys(), key=lambda k: hparam_results[k].get("test_f1", float("-inf")))


def find_repo_root(start: Path | None = None) -> Path:
    """Find repository root.

    Prefers the git root if available; otherwise searches upward for a folder named
    'hyperparameter_sweep' or 'real_fish'.
    """
    import subprocess

    candidates: list[Path] = []
    if start is not None:
        candidates.append(start)
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve())

    # 1) git root (best effort)
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(candidates[0].resolve()),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            gp = Path(r.stdout.strip())
            if gp.exists():
                return gp.resolve()
    except Exception:
        pass

    # 2) search for marker folders
    for base in candidates:
        base = base.resolve()
        for q in [base] + list(base.parents):
            if (q / "hyperparameter_sweep").exists() or (q / "real_fish").exists():
                return q

    return Path.cwd().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a subtype-specific lightning-action binary model.")
    parser.add_argument(
        "--project_root",
        type=Path,
        default=None,
        help="Repo root (auto-detected if omitted). Should contain 'hyperparameter_sweep/'.",
    )
    parser.add_argument("--variant", type=str, default="LP_with_cal_contour")
    parser.add_argument("--subtype", type=str, required=True, choices=["fighting", "hybrid", "ornamental", "wild"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "all"])
    parser.add_argument("--hparam_key", type=str, default=None, help="e.g. nh64_ly3_lg8_lr5e-04")
    parser.add_argument("--use_best_from_results", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=1000)
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--post_inference", action="store_true", help="Run post-training inference on training expts.")
    parser.add_argument("--output_tag", type=str, default=None, help="Optional extra suffix for output dir.")
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Write config.yaml with data_path relative to repo root (more portable).",
    )

    args = parser.parse_args()

    project_root = (args.project_root or find_repo_root()).resolve()
    sweep_root = project_root / "hyperparameter_sweep"

    data_path = sweep_root / "binary_data" / args.subtype / args.variant
    splits_file = data_path / "splits.yaml"
    splits = parse_splits(splits_file)

    if args.split == "train":
        expt_ids = splits["train"]
    else:
        expt_ids = splits["train"] + splits["test"] + splits.get("val", [])

    # resolve aliases for on-disk filenames (ornamental oR -> R)
    expt_ids = [resolve_session_alias(s, args.subtype) for s in expt_ids]

    if args.use_best_from_results:
        results_file = sweep_root / "results" / f"{args.subtype}_results.json"
        args.hparam_key = load_best_key(results_file)

    if not args.hparam_key:
        raise SystemExit("Provide --hparam_key or --use_best_from_results")

    h = parse_hparam_key(args.hparam_key)

    # input size for LP_with_cal_contour is 18 (as used in Week 14 notebook)
    input_size = 18
    output_size = 2

    config = build_binary_dtcn_config(
        data_path=data_path,
        expt_ids=expt_ids,
        input_size=input_size,
        output_size=output_size,
        num_hid_units=h["num_hid_units"],
        num_layers=h["num_layers"],
        num_lags=h["num_lags"],
        lr=h["lr"],
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
    )
    config["training"]["device"] = args.device
    if args.relative_paths:
        try:
            config["data"]["data_path"] = str(Path(config["data"]["data_path"]).resolve().relative_to(project_root))
        except Exception:
            # if relative conversion fails, keep absolute
            pass

    output_root = sweep_root / "models" / args.subtype
    out_dir_name = args.hparam_key
    if args.split == "all":
        out_dir_name = f"{out_dir_name}__all"
    if args.output_tag:
        out_dir_name = f"{out_dir_name}__{args.output_tag}"

    output_dir = output_root / out_dir_name
    save_config_yaml(config, output_dir)

    print("=" * 80)
    print(f"TRAIN SUBTYPE MODEL: {args.subtype} ({args.variant})")
    print(f"Split: {args.split}  |  Sessions: {len(expt_ids)}")
    print(f"Hyperparams: {args.hparam_key}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    model = Model.from_config(config)
    model.train(output_dir=output_dir, post_inference=args.post_inference)


if __name__ == "__main__":
    main()


