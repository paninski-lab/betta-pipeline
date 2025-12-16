#!/usr/bin/env python3
# coding: utf-8

import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_labels_onehot_csv(csv_path: str) -> np.ndarray:
    """
    Load one-hot labels stored in CSV and return class indices per frame.
    Assumes the CSV has an index column in col 0 (like your current format).
    """
    df = pd.read_csv(csv_path, index_col=0)
    return df.values.argmax(axis=1)


def iter_video_frames(video_path: str, max_frames: int | None = None):
    """
    Generator that yields (frame_idx, rgb_frame).
    Uses OpenCV read loop to avoid storing all frames in memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    i = 0
    while True:
        if max_frames is not None and i >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield i, rgb
        i += 1

    cap.release()


def run_inference_on_video(
    model: YOLO,
    video_path: str,
    device: str,
    max_frames: int | None = None,
) -> tuple[list[int], list[int], list[np.ndarray]]:
    """
    Returns:
      frame_indices: list[int]
      pred_labels:  list[int]
      probs_list:   list[np.ndarray]  shape (n_classes,)
    """
    frame_indices = []
    pred_labels = []
    probs_list = []

    for frame_idx, rgb in tqdm(
        iter_video_frames(video_path, max_frames=max_frames),
        desc=os.path.basename(video_path),
        unit="frame",
    ):
        res = model.predict(rgb, verbose=False, device=device, stream=False)[0]
        pred_labels.append(int(res.probs.top1))
        probs_list.append(res.probs.data.detach().cpu().numpy())
        frame_indices.append(frame_idx)

        # free references quickly (helpful on GPU runs)
        del rgb, res

    return frame_indices, pred_labels, probs_list


def save_predictions_csv(
    pred_csv_path: str,
    frame_indices: list[int],
    pred_labels: list[int],
    probs_list: list[np.ndarray],
    class_names: dict | list,
):
    n_classes = len(class_names)
    df_pred = pd.DataFrame(
        probs_list,
        columns=[f"prob_{class_names[i]}" for i in range(n_classes)],
    )
    df_pred.insert(0, "frame", frame_indices)
    df_pred.insert(1, "prediction", pred_labels)
    df_pred.to_csv(pred_csv_path, index=False)


def find_videos(video_dir: str) -> list[str]:
    """
    Find cropped videos recursively. Matches '*_cropped.mp4' by default.
    """
    matches = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith("_cropped.mp4"):
                matches.append(os.path.join(root, f))
    return sorted(matches)


def base_id_from_video(video_path: str) -> str:
    """
    Convert '/path/3558_robot_cropped.mp4' -> '3558_robot'
    """
    name = os.path.basename(video_path)
    if name.endswith("_cropped.mp4"):
        return name.replace("_cropped.mp4", "")
    return os.path.splitext(name)[0]


def label_path_for_base_id(label_dir: str, base_id: str) -> str:
    """
    Expected label naming: manual_scoring_{base_id}.csv
    """
    return os.path.join(label_dir, f"manual_scoring_{base_id}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO classification pipeline over cropped videos."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to a single *_cropped.mp4 video")
    group.add_argument("--video_dir", type=str, help="Directory containing *_cropped.mp4 videos")

    parser.add_argument("--model", type=str, required=True, help="Path to YOLO classification model (.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory (e.g. outputs/yolo_classification)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--label_dir", type=str, default=None, help="Optional: directory containing manual_scoring_{base_id}.csv")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional: limit frames per video (debug)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute predictions even if CSV exists")

    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)
    class_names = model.names
    n_classes = len(class_names)

    # Prepare outputs
    output_dir = ensure_dir(args.output_dir)
    pred_dir = ensure_dir(os.path.join(output_dir, "predictions"))
    summary_dir = ensure_dir(os.path.join(output_dir, "summary"))

    # Collect videos
    if args.video is not None:
        video_paths = [args.video]
    else:
        video_paths = find_videos(args.video_dir)

    if len(video_paths) == 0:
        raise RuntimeError("No videos found. Make sure you point to *_cropped.mp4 files.")

    # Storage (only if labels are provided)
    all_true = []
    all_pred = []
    per_video_stats = []

    for video_path in video_paths:
        base_id = base_id_from_video(video_path)
        pred_csv_path = os.path.join(pred_dir, f"predictions_{base_id}.csv")

        print(f"\n📂 [{base_id}] Starting...")

        # Load labels if available
        true_labels = None
        if args.label_dir is not None:
            lab_path = label_path_for_base_id(args.label_dir, base_id)
            if os.path.exists(lab_path):
                true_labels = load_labels_onehot_csv(lab_path)
                print(f"📄 [{base_id}] Loaded labels: {len(true_labels)} frames")
            else:
                print(f"⚠️  [{base_id}] Label file not found: {lab_path}")

        # Load cached predictions or run inference
        if os.path.exists(pred_csv_path) and not args.overwrite:
            print(f"📄 [{base_id}] Loading cached predictions: {pred_csv_path}")
            df_pred = pd.read_csv(pred_csv_path)
            pred_labels = df_pred["prediction"].astype(int).tolist()
            frame_indices = df_pred["frame"].astype(int).tolist()

        else:
            print(f"🔁 [{base_id}] Running inference...")
            frame_indices, pred_labels, probs_list = run_inference_on_video(
                model=model,
                video_path=video_path,
                device=args.device,
                max_frames=args.max_frames,
            )

            save_predictions_csv(
                pred_csv_path=pred_csv_path,
                frame_indices=frame_indices,
                pred_labels=pred_labels,
                probs_list=probs_list,
                class_names=class_names,
            )
            print(f"💾 [{base_id}] Saved predictions: {pred_csv_path}")

        # If we have labels, align and compute per-video stats
        if true_labels is not None:
            min_len = min(len(true_labels), len(pred_labels))
            y_true = true_labels[:min_len]
            y_pred = np.array(pred_labels[:min_len], dtype=int)

            all_true.extend(y_true.tolist())
            all_pred.extend(y_pred.tolist())

            tr = np.bincount(y_true, minlength=n_classes) / len(y_true)
            pr = np.bincount(y_pred, minlength=n_classes) / len(y_pred)

            per_video_stats.append({
                "base_id": base_id,
                "n_frames_eval": int(min_len),
                **{f"true_rate_{class_names[i]}": float(tr[i]) for i in range(n_classes)},
                **{f"pred_rate_{class_names[i]}": float(pr[i]) for i in range(n_classes)},
            })

            print(f"✅ [{base_id}] Done (evaluated on {min_len} frames).")
        else:
            print(f"✅ [{base_id}] Done (predictions only).")

    # Save summary
    if len(per_video_stats) > 0:
        df_stats = pd.DataFrame(per_video_stats)
        stats_path = os.path.join(summary_dir, "per_video_rates.csv")
        df_stats.to_csv(stats_path, index=False)
        print(f"\n💾 Saved per-video summary: {stats_path}")

    # Global report (only if labels provided)
    if args.label_dir is not None and len(all_true) > 0 and len(all_pred) > 0:
        all_true = np.array(all_true, dtype=int)
        all_pred = np.array(all_pred, dtype=int)

        if SKLEARN_AVAILABLE:
            report = classification_report(all_true, all_pred, target_names=[class_names[i] for i in range(n_classes)], digits=4)
            cm = confusion_matrix(all_true, all_pred, labels=list(range(n_classes)))

            report_path = os.path.join(summary_dir, "classification_report.txt")
            cm_path = os.path.join(summary_dir, "confusion_matrix.csv")

            with open(report_path, "w") as f:
                f.write(report)

            pd.DataFrame(cm, index=[class_names[i] for i in range(n_classes)], columns=[class_names[i] for i in range(n_classes)]).to_csv(cm_path)

            print(f"\n📄 Saved classification report: {report_path}")
            print(f"📄 Saved confusion matrix: {cm_path}")
        else:
            print("\n⚠️ scikit-learn not available. Skipping classification_report and confusion_matrix.")

    # Save run metadata
    meta = {
        "model_path": args.model,
        "device": args.device,
        "n_videos": len(video_paths),
        "videos_sample": video_paths[:5],
        "label_dir": args.label_dir,
        "max_frames": args.max_frames,
        "overwrite": args.overwrite,
    }
    meta_path = os.path.join(summary_dir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n🧾 Saved run metadata: {meta_path}")

    print("\n🏁 Done.")


if __name__ == "__main__":
    main()
