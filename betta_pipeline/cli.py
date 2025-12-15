import argparse
import glob
import os

from betta_pipeline.train import lp_train
from betta_pipeline.feature_generation import feature_generation


def main():
    parser = argparse.ArgumentParser(
        prog="betta",
        description="Betta fish behavior analysis pipeline",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -------------------------------------------------
    # train subcommand
    # -------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Run Lightning Pose inference on videos",
    )
    train_parser.add_argument(
        "--video-folder",
        required=True,
        help="Directory containing input .mp4 videos",
    )
    train_parser.add_argument(
        "--output-folder",
        required=True,
        help="Directory to save prediction CSV files",
    )
    train_parser.add_argument(
        "--cfg-file",
        required=True,
        help="Path to Lightning Pose config YAML",
    )
    train_parser.add_argument(
        "--ckpt-file",
        required=True,
        help="Path to Lightning Pose checkpoint (.ckpt)",
    )

    # -------------------------------------------------
    # feature-generation subcommand
    # -------------------------------------------------
    feat_parser = subparsers.add_parser(
        "feature-generation",
        help="Generate behavioral features from pose files",
    )
    feat_parser.add_argument(
        "--input",
        required=True,
        help="Folder containing pose files (.csv or .h5)",
    )
    feat_parser.add_argument(
        "--output",
        required=True,
        help="Folder to save generated feature CSVs",
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Dispatch
    # -------------------------------------------------
    if args.command == "train":
        lp_train(
            video_folder=args.video_folder,
            output_folder=args.output_folder,
            cfg_file=args.cfg_file,
            ckpt_file=args.ckpt_file,
        )

    elif args.command == "feature-generation":
        os.makedirs(args.output, exist_ok=True)

        files = (
            glob.glob(os.path.join(args.input, "*.csv"))
            + glob.glob(os.path.join(args.input, "*.h5"))
        )

        if not files:
            raise RuntimeError(
                f"No .csv or .h5 files found in {args.input}"
            )

        feature_generation(files, args.output)