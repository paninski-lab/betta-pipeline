# Fish Tracking Pipeline with Gaussian Smoothing

A computer vision pipeline for automated fish trajectory tracking and analysis using YOLOv8 object detection with advanced smoothing algorithms.

## Overview

This pipeline implements an automated fish tracking system designed to:
- Detect and track fish movements in video footage using YOLOv8
- Apply median filtering and Gaussian smoothing for trajectory refinement
- Identify and filter outlier detections through trajectory deviation analysis and robot fish overlap detection
- Generate cropped, centered video output following the fish's smoothed trajectory
- Provide comprehensive visualization and analysis tools

## Key Features

- **Flexible Processing**: Supports both GPU (CUDA) and CPU processing
- **Advanced Smoothing**: Two-stage filtering combining median and Gaussian smoothing for robust trajectory estimation
- **Dual Outlier Detection**: 
  - Trajectory deviation analysis to identify abnormal position jumps
  - Robot fish overlap detection using Intersection over Union (IoU) metrics
- **Interpolation**: Cubic spline interpolation for handling missing frames in detection coverage
- **Single Fish Tracking**: Confidence-based selection when multiple fish are detected per frame
- **Comprehensive Output**: Annotated videos, trajectory plots, and detailed CSV analytics

---

## Environment

This pipeline expects the YOLO Conda environment defined in the repository root.

From the repository root:
```
conda activate yolo-env
```
---

## Usage

### Basic Command

```bash
python median_gaussian.py \
    --video path/to/video.mp4 \
    --video_name experiment_01 \
    --model path/to/model.pt \
    --output ./output
```

### Full Command with All Parameters

```bash
python median_gaussian.py \
    --video data/videos/tank_recording.mp4 \
    --video_name tank_01_trial_1 \
    --model models/fish_detector.pt \
    --output ./output \
    --conf 0.6 \
    --box_size 200 \
    --median_kernel 5 \
    --gaussian_sigma 3 \
    --deviation_threshold 50 \
    --iou_threshold 0.3 \
    --batch_size 16 \
    --device cuda
```

## Pipeline Architecture

```
Input Video → YOLO Detection → Primary Fish Selection → Smoothing → Outlier Detection → Cropping → Output
                    ↓                    ↓                   ↓              ↓               ↓
              Batch Processing   Confidence-based      Median +      Deviation +    Interpolation
              (GPU or CPU)       Multi-fish Handling   Gaussian      Robot IoU      for Gaps
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--video` | Required | Path to input video file |
| `--video_name` | Required | Name identifier for output files |
| `--model` | Required | Path to trained YOLO model (.pt) |
| `--output` | Required | Output directory path |
| `--conf` | 0.6 | Detection confidence threshold (0.0-1.0) |
| `--box_size` | 200 | Crop box size in pixels |
| `--median_kernel` | 5 | Median filter window size (3, 5, 7, 9, 11) |
| `--gaussian_sigma` | 3 | Gaussian smoothing strength (1-7) |
| `--deviation_threshold` | 50 | Maximum trajectory deviation in pixels |
| `--iou_threshold` | 0.3 | IoU threshold for robot overlap detection |
| `--batch_size` | 16 | Number of frames to process in parallel |
| `--device` | cuda | Processing device (cuda or cpu) |

## Output Structure

```
output/
├── videos/
│   ├── experiment_01_detected.mp4    # Original video with detection annotations
│   └── experiment_01_cropped.mp4     # Centered, cropped output following fish
├── csv/
│   ├── experiment_01_detections.csv           # Raw detection data
│   ├── experiment_01_outliers.csv             # Flagged anomalies
│   └── experiment_01_smoothed_trajectory.csv  # Processed trajectory data
└── plots/
    ├── experiment_01_trajectory_axes.png  # X and Y coordinates over time
    └── experiment_01_trajectory_xy.png    # 2D spatial trajectory visualization
```

## Technical Details

### Smoothing Algorithm

The pipeline uses a two-stage smoothing approach:

1. **Median Filtering**: Removes sudden spikes and noise from raw detections
2. **Gaussian Smoothing**: Creates smooth, continuous trajectory with configurable sigma
3. **Cubic Interpolation**: Fills gaps in detection coverage for complete video cropping

### Outlier Detection

The pipeline identifies outliers using two methods:

1. **Trajectory Deviation Analysis**: Detects fish positions that deviate beyond a threshold distance from the smoothed trajectory, indicating potential false detections or tracking errors
2. **Robot Fish Overlap Detection**: Calculates Intersection over Union (IoU) between fish and robot fish bounding boxes to identify frames where detections may be contaminated by robot interference

### Multi-Fish Handling

When multiple fish are detected in a single frame, the system automatically selects the detection with the highest confidence score, ensuring consistent tracking of a single target throughout the video.

## Output Files Description

### CSV Files

**detections.csv**: Contains all raw YOLO detections
- `frame`: Frame number in video
- `class`: Detection class (fish or robot_fish)
- `conf`: Confidence score (0-1)
- `x1, y1, x2, y2`: Bounding box coordinates
- `width, height`: Bounding box dimensions

**smoothed_trajectory.csv**: Contains processed trajectory data
- `frame`: Frame number
- `cx, cy`: Raw center coordinates
- `cx_smooth, cy_smooth`: Smoothed center coordinates
- `is_outlier`: Boolean flag indicating outlier status
- `deviation`: Distance from smoothed trajectory (pixels)

**outliers.csv**: Contains details on detected outliers
- `frame`: Frame number with outlier
- `outlier_reason`: Explanation (trajectory deviation or robot overlap)
- `cx, cy`: Position of outlier
- `deviation`: Deviation magnitude if applicable

### Plot Files

**trajectory_axes.png**: Shows X and Y coordinates separately over time
- Blue line represents raw detections
- Orange line represents smoothed trajectory
- Red X markers indicate detected outliers

**trajectory_xy.png**: Shows 2D spatial trajectory
- Blue line represents raw movement path
- Orange line represents smoothed path
- Red X markers indicate outlier locations
- Y-axis is inverted to match image coordinates
