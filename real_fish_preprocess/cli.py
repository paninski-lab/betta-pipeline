import argparse
import glob
import os

from real_fish_preprocess.pose_predict import lp_predict
from real_fish_preprocess.feature_generation import feature_generation


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
    # predict subcommand
    # -------------------------------------------------
    predict_parser = subparsers.add_parser(
        "pose-predict",
        help="Run Lightning Pose inference on videos",
    )

    predict_parser.add_argument(
        "--video",
        help="Path to a single input .mp4 video",
    )

    predict_parser.add_argument(
        "--video-folder",
        help="Directory containing input .mp4 videos",
    )
    predict_parser.add_argument(
        "--output-folder",
        required=True,
        help="Directory to save prediction CSV files",
    )
    predict_parser.add_argument(
        "--cfg-file",
        required=True,
        help="Path to Lightning Pose config YAML",
    )
    predict_parser.add_argument(
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
    if args.command == "pose-predict":
       if bool(args.video) == bool(args.video_folder):
            raise RuntimeError("Provide exactly one of --video or --video-folder")

       lp_predict(
            video=args.video,
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

    else:
        raise RuntimeError(f"Unknown command: {args.command}")

