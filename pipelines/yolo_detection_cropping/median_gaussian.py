#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import torch


# ============================================================
# Utility
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================
# Outlier Detection
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.
    Each box is [x1, y1, x2, y2].
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def detect_outliers_and_smooth(csv_path, median_kernel=5, gaussian_sigma=3, 
                               deviation_threshold=50, iou_threshold=0.3):
    """
    Detect outlier fish detections and apply Gaussian smoothing.
    
    Parameters:
    -----------
    csv_path : str
        Path to detections CSV
    median_kernel : int
        Median filter window size (must be odd: 3, 5, 7, 9, 11)
    gaussian_sigma : float
        Gaussian smoothing strength (1=light, 3=balanced, 5+=heavy)
    deviation_threshold : float
        Max allowed deviation (pixels) from smoothed trajectory
    iou_threshold : float
        IoU threshold for robot overlap detection
    
    Returns:
    --------
    DataFrame with smoothed trajectories and outlier flags.
    """
    df = pd.read_csv(csv_path)
    
    # Separate fish and robot_fish detections
    fish_df = df[df["class"] == "fish"].copy().reset_index(drop=True)
    robot_df = df[df["class"] == "robot_fish"].copy()
    
    # Calculate fish centers
    fish_df["cx"] = (fish_df["x1"] + fish_df["x2"]) / 2
    fish_df["cy"] = (fish_df["y1"] + fish_df["y2"]) / 2
    
    # Initialize outlier flag
    fish_df["is_outlier"] = False
    fish_df["outlier_reason"] = ""
    
    # ========================================
    # Median + Gaussian Smoothing
    # ========================================
    
    if len(fish_df) >= median_kernel:
        # Stage 1: Apply median filter to remove spikes
        cx_median = medfilt(fish_df["cx"].values, kernel_size=median_kernel)
        cy_median = medfilt(fish_df["cy"].values, kernel_size=median_kernel)
        
        # Stage 2: Apply Gaussian smoothing for smooth trajectory
        cx_smooth = gaussian_filter1d(cx_median, sigma=gaussian_sigma)
        cy_smooth = gaussian_filter1d(cy_median, sigma=gaussian_sigma)
        
        # Store smoothed trajectory
        fish_df["cx_smooth"] = cx_smooth
        fish_df["cy_smooth"] = cy_smooth
        
        # Calculate deviation from smoothed trajectory
        fish_df["deviation"] = np.sqrt(
            (fish_df["cx"] - cx_smooth)**2 + 
            (fish_df["cy"] - cy_smooth)**2
        )
        
        # Mark points that deviate too much from smoothed trajectory
        deviation_mask = fish_df["deviation"] > deviation_threshold
        fish_df.loc[deviation_mask, "is_outlier"] = True
        fish_df.loc[deviation_mask, "outlier_reason"] = \
            fish_df.loc[deviation_mask].apply(
                lambda row: f"deviation_{row['deviation']:.1f}px", axis=1
            )
    
    # ========================================
    # Robot Overlap Detection
    # ========================================
    
    for idx, fish_row in fish_df.iterrows():
        frame_num = fish_row["frame"]
        fish_box = [fish_row["x1"], fish_row["y1"], fish_row["x2"], fish_row["y2"]]
        
        # Get all robot_fish in this frame
        robots_in_frame = robot_df[robot_df["frame"] == frame_num]
        
        for _, robot_row in robots_in_frame.iterrows():
            robot_box = [robot_row["x1"], robot_row["y1"], robot_row["x2"], robot_row["y2"]]
            iou = calculate_iou(fish_box, robot_box)
            
            if iou > iou_threshold:
                fish_df.loc[idx, "is_outlier"] = True
                reason = fish_df.loc[idx, "outlier_reason"]
                fish_df.loc[idx, "outlier_reason"] = \
                    f"{reason};robot_overlap_IoU={iou:.2f}" if reason else f"robot_overlap_IoU={iou:.2f}"
    
    # Print outlier statistics
    n_outliers = fish_df["is_outlier"].sum()
    n_total = len(fish_df)
    print(f"\n📊 Outlier Detection Results:")
    print(f"   Total fish detections: {n_total}")
    print(f"   Outliers detected: {n_outliers} ({100*n_outliers/n_total:.1f}%)")
    
    if n_outliers > 0:
        print(f"\n   Outlier breakdown:")
        deviation_outliers = fish_df[fish_df["outlier_reason"].str.contains("deviation", na=False)]
        robot_outliers = fish_df[fish_df["outlier_reason"].str.contains("robot", na=False)]
        print(f"   - Trajectory deviations: {len(deviation_outliers)}")
        print(f"   - Robot overlap: {len(robot_outliers)}")
        
        if len(deviation_outliers) > 0:
            avg_dev = fish_df[fish_df["outlier_reason"].str.contains("deviation", na=False)]["deviation"].mean()
            print(f"   - Average deviation of outliers: {avg_dev:.1f}px")
    
    return fish_df

def select_primary_fish(fish_df):
    """
    When multiple fish are detected per frame, select the fish with highest confidence.
    This is simpler and more reliable than tracking by position.
    
    Returns: DataFrame with one fish per frame
    """
    if len(fish_df) == 0:
        return fish_df
    
    # Calculate centers for all detections
    fish_df["cx"] = (fish_df["x1"] + fish_df["x2"]) / 2
    fish_df["cy"] = (fish_df["y1"] + fish_df["y2"]) / 2
    
    # Group by frame and select highest confidence detection
    selected_rows = []
    grouped = fish_df.groupby("frame")
    
    for frame_num, group in grouped:
        if len(group) == 1:
            # Only one fish - easy choice
            selected_rows.append(group.iloc[0])
        else:
            # Multiple fish - select the one with highest confidence
            selected = group.loc[group["conf"].idxmax()]
            selected_rows.append(selected)
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    
    # Report statistics
    total_detections = len(fish_df)
    frames_with_multiple = len([g for _, g in grouped if len(g) > 1])
    
    if frames_with_multiple > 0:
        print(f"📊 Multiple fish per frame detected:")
        print(f"   Total fish detections: {total_detections}")
        print(f"   Unique frames: {len(result_df)}")
        print(f"   Frames with multiple fish: {frames_with_multiple}")
        print(f"   Selected highest confidence detection per frame")
        
        # Show confidence statistics for multi-fish frames
        multi_fish_frames = [g for _, g in grouped if len(g) > 1]
        if multi_fish_frames:
            avg_confs = [g["conf"].mean() for g in multi_fish_frames]
            max_confs = [g["conf"].max() for g in multi_fish_frames]
            print(f"   Average confidence in multi-fish frames: {np.mean(avg_confs):.3f}")
            print(f"   Average of selected (max) confidences: {np.mean(max_confs):.3f}")
    
    return result_df


def detect_outliers_and_smooth_single_fish(csv_path, median_kernel=5, gaussian_sigma=3, 
                                          deviation_threshold=50, iou_threshold=0.3):
    """
    Modified version that ensures only one fish is tracked per frame.
    """
    df = pd.read_csv(csv_path)
    
    # Separate fish and robot_fish detections
    fish_df = df[df["class"] == "fish"].copy().reset_index(drop=True)
    robot_df = df[df["class"] == "robot_fish"].copy()
    
    # SELECT PRIMARY FISH (one per frame)
    fish_df = select_primary_fish(fish_df)
    
    # Now proceed with smoothing (cx, cy already calculated in select_primary_fish)
    fish_df["is_outlier"] = False
    fish_df["outlier_reason"] = ""
    
    # Median + Gaussian Smoothing
    if len(fish_df) >= median_kernel:
        cx_median = medfilt(fish_df["cx"].values, kernel_size=median_kernel)
        cy_median = medfilt(fish_df["cy"].values, kernel_size=median_kernel)
        
        cx_smooth = gaussian_filter1d(cx_median, sigma=gaussian_sigma)
        cy_smooth = gaussian_filter1d(cy_median, sigma=gaussian_sigma)
        
        fish_df["cx_smooth"] = cx_smooth
        fish_df["cy_smooth"] = cy_smooth
        
        fish_df["deviation"] = np.sqrt(
            (fish_df["cx"] - cx_smooth)**2 + 
            (fish_df["cy"] - cy_smooth)**2
        )
        
        deviation_mask = fish_df["deviation"] > deviation_threshold
        fish_df.loc[deviation_mask, "is_outlier"] = True
        fish_df.loc[deviation_mask, "outlier_reason"] = \
            fish_df.loc[deviation_mask].apply(
                lambda row: f"deviation_{row['deviation']:.1f}px", axis=1
            )
    
    # Robot Overlap Detection
    for idx, fish_row in fish_df.iterrows():
        frame_num = fish_row["frame"]
        fish_box = [fish_row["x1"], fish_row["y1"], fish_row["x2"], fish_row["y2"]]
        
        robots_in_frame = robot_df[robot_df["frame"] == frame_num]
        
        for _, robot_row in robots_in_frame.iterrows():
            robot_box = [robot_row["x1"], robot_row["y1"], robot_row["x2"], robot_row["y2"]]
            
            from scipy.spatial.distance import cdist
            # Simple IoU calculation
            x1_inter = max(fish_box[0], robot_box[0])
            y1_inter = max(fish_box[1], robot_box[1])
            x2_inter = min(fish_box[2], robot_box[2])
            y2_inter = min(fish_box[3], robot_box[3])
            
            if x2_inter > x1_inter and y2_inter > y1_inter:
                inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                fish_area = (fish_box[2] - fish_box[0]) * (fish_box[3] - fish_box[1])
                robot_area = (robot_box[2] - robot_box[0]) * (robot_box[3] - robot_box[1])
                union_area = fish_area + robot_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                
                if iou > iou_threshold:
                    fish_df.loc[idx, "is_outlier"] = True
                    reason = fish_df.loc[idx, "outlier_reason"]
                    fish_df.loc[idx, "outlier_reason"] = \
                        f"{reason};robot_overlap_IoU={iou:.2f}" if reason else f"robot_overlap_IoU={iou:.2f}"
    
    # Print statistics
    n_outliers = fish_df["is_outlier"].sum()
    n_total = len(fish_df)
    print(f"\n📊 Outlier Detection Results:")
    print(f"   Total fish detections: {n_total}")
    print(f"   Outliers detected: {n_outliers} ({100*n_outliers/n_total:.1f}%)")
    
    if n_outliers > 0:
        print(f"\n   Outlier breakdown:")
        deviation_outliers = fish_df[fish_df["outlier_reason"].str.contains("deviation", na=False)]
        robot_outliers = fish_df[fish_df["outlier_reason"].str.contains("robot", na=False)]
        print(f"   - Trajectory deviations: {len(deviation_outliers)}")
        print(f"   - Robot overlap: {len(robot_outliers)}")
        
        if len(deviation_outliers) > 0:
            avg_dev = fish_df[fish_df["outlier_reason"].str.contains("deviation", na=False)]["deviation"].mean()
            print(f"   - Average deviation of outliers: {avg_dev:.1f}px")
    
    return fish_df

def process_trajectory(csv_path, output_dir, video_name, 
                      median_kernel=5, gaussian_sigma=3, 
                      deviation_threshold=50, iou_threshold=0.3):
    """
    Detect outliers and apply Gaussian smoothing.
    Saves results to CSV files.
    """
    # Detect outliers and smooth
    fish_df = detect_outliers_and_smooth_single_fish(
        csv_path, 
        median_kernel=median_kernel,
        gaussian_sigma=gaussian_sigma,
        deviation_threshold=deviation_threshold, 
        iou_threshold=iou_threshold
    )
    
    # Save results
    csv_dir = ensure_dir(os.path.join(output_dir, "csv"))
    
    # Save outlier frames
    outliers_csv_path = os.path.join(csv_dir, f"{video_name}_outliers.csv")
    outlier_cols = ["frame", "outlier_reason", "cx", "cy"]
    if "deviation" in fish_df.columns:
        outlier_cols.append("deviation")
    outlier_frames = fish_df[fish_df["is_outlier"] == True][outlier_cols].copy()
    outlier_frames.to_csv(outliers_csv_path, index=False)
    print(f"✅ Saved outlier frames → {outliers_csv_path}")
    
    # Save smoothed trajectories
    trajectory_csv_path = os.path.join(csv_dir, f"{video_name}_smoothed_trajectory.csv")
    trajectory_cols = ["frame", "cx", "cy", "cx_smooth", "cy_smooth", "is_outlier"]
    if "deviation" in fish_df.columns:
        trajectory_cols.append("deviation")
    trajectory_df = fish_df[trajectory_cols].copy()
    trajectory_df.to_csv(trajectory_csv_path, index=False)
    print(f"✅ Saved smoothed trajectory → {trajectory_csv_path}")
    
    # Plot
    plot_dir = os.path.join(output_dir, "plots")
    plot_fish_trajectory(fish_df, save_dir=plot_dir, video_name=video_name)
    
    return fish_df


# ============================================================
# Plotting
# ============================================================

def plot_fish_trajectory(fish_df, save_dir=None, video_name="plot"):
    """
    Plot raw vs Gaussian smoothed trajectory, highlighting outliers.
    """
    os.makedirs(save_dir, exist_ok=True)

    frames = fish_df["frame"].values
    outlier_frames = fish_df[fish_df["is_outlier"] == True]["frame"].values
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for i, axis in enumerate(["cx", "cy"]):
        raw_col = axis
        smooth_col = f"{axis}_smooth"
        ax[i].plot(frames, fish_df[raw_col], label="Raw", color="blue", alpha=0.5)
        ax[i].plot(frames, fish_df[smooth_col], label="Gaussian Smoothed", color="orange", linewidth=2)
        
        # Mark outliers
        if len(outlier_frames) > 0:
            outlier_data = fish_df[fish_df["is_outlier"] == True]
            ax[i].scatter(outlier_data["frame"], outlier_data[raw_col], 
                         color="red", s=50, marker="x", label="Outliers", zorder=5)
        
        ax[i].set_ylabel(axis)
        ax[i].legend()
        ax[i].grid(alpha=0.3)

    ax[-1].set_xlabel("Frame")
    plt.suptitle(f"Fish Trajectory — {video_name}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{video_name}_trajectory_axes.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved per-axis plot → {save_path}")

    # 2D trajectory
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.plot(fish_df["cx"], fish_df["cy"], color="blue", alpha=0.5, label="Raw", linewidth=1)
    ax2.plot(fish_df["cx_smooth"], fish_df["cy_smooth"], color="orange", linewidth=2, label="Gaussian Smoothed")
    
    # Mark outliers in 2D
    if len(outlier_frames) > 0:
        outlier_data = fish_df[fish_df["is_outlier"] == True]
        ax2.scatter(outlier_data["cx"], outlier_data["cy"], 
                   color="red", s=100, marker="x", label="Outliers", zorder=5, linewidths=2)
    
    ax2.set_title(f"2D Fish Trajectory — {video_name}")
    ax2.set_xlabel("cx")
    ax2.set_ylabel("cy")
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(alpha=0.3)

    save_path_xy = os.path.join(save_dir, f"{video_name}_trajectory_xy.png")
    plt.savefig(save_path_xy, dpi=300)
    plt.close(fig2)
    print(f"✅ Saved 2D XY plot → {save_path_xy}")


# ============================================================
# YOLO Detection with GPU Batch Processing
# ============================================================

def run_detection(video_path, model, output_dir, video_name, conf=0.6, batch_size=16, device='cuda'):
    """
    Run YOLO detection with GPU batch processing for significant speedup.
    
    Args:
        batch_size: Number of frames to process in parallel (adjust based on GPU memory)
        device: 'cuda' for GPU, 'cpu' for CPU
    """
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        print(f"✅ Using device: {device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    videos_dir = ensure_dir(os.path.join(output_dir, "videos"))
    csv_dir = ensure_dir(os.path.join(output_dir, "csv"))

    detected_video_path = os.path.join(videos_dir, f"{video_name}_detected.mp4")
    csv_path = os.path.join(csv_dir, f"{video_name}_detections.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_detected = cv2.VideoWriter(detected_video_path, fourcc, fps, (w, h))

    detections = []
    frame_buffer = []
    frame_indices = []
    
    pbar = tqdm(total=total_frames, desc="Running detection (batched)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames in buffer
            if frame_buffer:
                results = model(frame_buffer, conf=conf, verbose=False, device=device)
                
                for result, idx in zip(results, frame_indices):
                    annotated = result.plot()
                    out_detected.write(annotated)
                    
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        label = model.names[cls]
                        conf_score = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        width = x2 - x1
                        height = y2 - y1
                        detections.append([idx, label, conf_score, x1, y1, x2, y2, width, height])
                
                pbar.update(len(frame_buffer))
            break
        
        frame_buffer.append(frame)
        frame_indices.append(frame_idx)
        frame_idx += 1
        
        # Process batch when buffer is full
        if len(frame_buffer) >= batch_size:
            results = model(frame_buffer, conf=conf, verbose=False, device=device)
            
            for result, idx in zip(results, frame_indices):
                annotated = result.plot()
                out_detected.write(annotated)
                
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf_score = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    width = x2 - x1
                    height = y2 - y1
                    detections.append([idx, label, conf_score, x1, y1, x2, y2, width, height])
            
            pbar.update(len(frame_buffer))
            frame_buffer = []
            frame_indices = []

    pbar.close()
    cap.release()
    out_detected.release()

    df = pd.DataFrame(detections, columns=["frame", "class", "conf", "x1", "y1", "x2", "y2", "width", "height"])
    df.to_csv(csv_path, index=False)

    return csv_path, detected_video_path, fps, (w, h)


# ============================================================
# Optimized Cropping with Gaussian Smoothed Trajectory
# ============================================================

# def crop_video(video_path, fish_df, output_dir, video_name, fps, orig_size,
#                box_size=200, save_cropped=True):
#     """
#     Crop video around fish position using ONLY Gaussian smoothed trajectory.
#     """
#     cap = cv2.VideoCapture(video_path)
#     w, h = orig_size

#     videos_dir = ensure_dir(os.path.join(output_dir, "videos"))
#     cropped_video_path = os.path.join(videos_dir, f"{video_name}_cropped.mp4")

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out_cropped = cv2.VideoWriter(cropped_video_path, fourcc, fps, (box_size, box_size)) if save_cropped else None
    
#     # Create lookup dictionary for fast frame access
#     frame_to_coords = {}
#     for _, row in fish_df.iterrows():
#         frame_num = int(row["frame"])
#         cx_smooth = int(row["cx_smooth"])
#         cy_smooth = int(row["cy_smooth"])
#         frame_to_coords[frame_num] = (cx_smooth, cy_smooth)
    
#     pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Cropping video")
#     frame_idx = 0
#     half = box_size // 2

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Get smoothed coordinates for this frame
#         if frame_idx in frame_to_coords:
#             cx, cy = frame_to_coords[frame_idx]
#         else:
#             # Skip frames without detections (shouldn't happen often)
#             pbar.update(1)
#             frame_idx += 1
#             continue
        
#         # Create fixed-size crop box centered on smoothed position
#         x1, y1 = max(0, cx - half), max(0, cy - half)
#         x2, y2 = min(w, cx + half), min(h, cy + half)
#         cropped = frame[y1:y2, x1:x2]
        
#         # Handle edge case where crop is smaller than box_size
#         if cropped.shape[0] > 0 and cropped.shape[1] > 0:
#             cropped_resized = cv2.resize(cropped, (box_size, box_size))
#         else:
#             cropped_resized = np.zeros((box_size, box_size, 3), dtype=np.uint8)

#         if out_cropped:
#             out_cropped.write(cropped_resized)

#         frame_idx += 1
#         pbar.update(1)

#     pbar.close()
#     cap.release()
#     if out_cropped:
#         out_cropped.release()
    
#     print(f"✅ Cropped video saved → {cropped_video_path}")

def crop_video(video_path, fish_df, output_dir, video_name, fps, orig_size,
               box_size=200, save_cropped=True):
    """
    Crop video around fish position using ONLY Gaussian smoothed trajectory.
    Uses cubic spline interpolation on smoothed trajectory for missing frames.
    """
    from scipy.interpolate import interp1d
    
    cap = cv2.VideoCapture(video_path)
    w, h = orig_size
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    videos_dir = ensure_dir(os.path.join(output_dir, "videos"))
    cropped_video_path = os.path.join(videos_dir, f"{video_name}_cropped.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_cropped = cv2.VideoWriter(cropped_video_path, fourcc, fps, (box_size, box_size)) if save_cropped else None
    
    # Get all smoothed coordinates
    frames = fish_df["frame"].values
    cx_smooth = fish_df["cx_smooth"].values
    cy_smooth = fish_df["cy_smooth"].values
    
    if len(frames) == 0:
        print("⚠️  No detections found, cannot crop video")
        cap.release()
        if out_cropped:
            out_cropped.release()
        return
    
    # Create interpolation functions from smoothed trajectory
    # Use cubic spline for smooth interpolation that respects trajectory
    min_frame = int(frames.min())
    max_frame = int(frames.max())
    
    if len(frames) >= 4:
        # Cubic spline requires at least 4 points
        interp_cx = interp1d(frames, cx_smooth, kind='cubic', fill_value='extrapolate')
        interp_cy = interp1d(frames, cy_smooth, kind='cubic', fill_value='extrapolate')
    else:
        # Fall back to linear for very few detections
        interp_cx = interp1d(frames, cx_smooth, kind='linear', fill_value='extrapolate')
        interp_cy = interp1d(frames, cy_smooth, kind='linear', fill_value='extrapolate')
    
    # Generate coordinates for all frames in detection range
    interpolated_coords = {}
    
    for frame_idx in range(total_frames):
        if min_frame <= frame_idx <= max_frame:
            # Interpolate within detection range
            cx = int(np.clip(interp_cx(frame_idx), 0, w))
            cy = int(np.clip(interp_cy(frame_idx), 0, h))
            interpolated_coords[frame_idx] = (cx, cy)
        elif frame_idx < min_frame:
            # Before first detection: use first position
            interpolated_coords[frame_idx] = (int(cx_smooth[0]), int(cy_smooth[0]))
        else:
            # After last detection: use last position
            interpolated_coords[frame_idx] = (int(cx_smooth[-1]), int(cy_smooth[-1]))
    
    # Count interpolated frames
    detected_frames = set(frames)
    n_interpolated = len([f for f in range(total_frames) if f not in detected_frames])
    if n_interpolated > 0:
        print(f"📊 Interpolation statistics:")
        print(f"   Total frames: {total_frames}")
        print(f"   Detected frames: {len(detected_frames)}")
        print(f"   Interpolated frames: {n_interpolated} ({100*n_interpolated/total_frames:.1f}%)")
        print(f"   Detection range: frames {min_frame} to {max_frame}")
    
    pbar = tqdm(total=total_frames, desc="Cropping video (with interpolation)")
    frame_idx = 0
    half = box_size // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get coordinates (detected or interpolated)
        if frame_idx in interpolated_coords:
            cx, cy = interpolated_coords[frame_idx]
        else:
            # Should never happen with proper interpolation
            print(f"⚠️  Warning: No coordinates for frame {frame_idx}")
            pbar.update(1)
            frame_idx += 1
            continue
        
        # Create fixed-size crop box centered on position
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        cropped = frame[y1:y2, x1:x2]
        
        # Handle edge case where crop is smaller than box_size
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            cropped_resized = cv2.resize(cropped, (box_size, box_size))
        else:
            cropped_resized = np.zeros((box_size, box_size, 3), dtype=np.uint8)

        if out_cropped:
            out_cropped.write(cropped_resized)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if out_cropped:
        out_cropped.release()
    
    print(f"✅ Cropped video saved → {cropped_video_path}")


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(video_path, model_path, output_dir, video_name,
                 conf=0.6, box_size=200, 
                 median_kernel=5, gaussian_sigma=3, deviation_threshold=50,
                 iou_threshold=0.3, batch_size=16, device='cuda'):
    """
    Run complete pipeline with GPU acceleration and Gaussian smoothing only.
    
    Args:
        median_kernel: Median filter window (3, 5, 7, 9, 11)
        gaussian_sigma: Gaussian smoothing strength (1-7)
        deviation_threshold: Max pixels from smoothed trajectory
        batch_size: Frames to process in parallel
        device: 'cuda' for GPU, 'cpu' for CPU
    """
    print(f"\n{'='*60}")
    print(f"Starting Pipeline: {video_name}")
    print(f"{'='*60}")
    
    model = YOLO(model_path)

    # Step 1: Detection with batch processing
    print("\n[1/3] Running YOLO Detection...")
    csv_path, detected_video_path, fps, orig_size = run_detection(
        video_path, model, output_dir, video_name, conf=conf,
        batch_size=batch_size, device=device
    )

    # Step 2: Gaussian smoothing only
    print("\n[2/3] Applying Gaussian Smoothing...")
    fish_df = process_trajectory(
        csv_path, output_dir, video_name,
        median_kernel=median_kernel, gaussian_sigma=gaussian_sigma,
        deviation_threshold=deviation_threshold, iou_threshold=iou_threshold
    )

    # Step 3: Crop using Gaussian smoothed trajectory
    print("\n[3/3] Cropping Video...")
    crop_video(video_path, fish_df, output_dir, video_name, fps, orig_size, box_size)
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline Complete: {video_name}")
    print(f"{'='*60}\n")


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU-accelerated fish tracking with Gaussian smoothing")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--video_name", type=str, required=True, help="Name for outputs")
    parser.add_argument("--model", type=str, required=True, help="YOLO model path (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--box_size", type=int, default=200, help="Crop box size around target")
    parser.add_argument("--median_kernel", type=int, default=5, help="Median filter window (3,5,7,9,11)")
    parser.add_argument("--gaussian_sigma", type=float, default=3, help="Gaussian smoothing sigma (1-7)")
    parser.add_argument("--deviation_threshold", type=float, default=50, help="Max deviation from smooth trajectory (pixels)")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for robot overlap")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for GPU processing")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help="Device for inference")
    args = parser.parse_args()

    run_pipeline(
        args.video,
        args.model,
        args.output,
        args.video_name,
        conf=args.conf,
        box_size=args.box_size,
        median_kernel=args.median_kernel,
        gaussian_sigma=args.gaussian_sigma,
        deviation_threshold=args.deviation_threshold,
        iou_threshold=args.iou_threshold,
        batch_size=args.batch_size,
        device=args.device
    )