# Betta Pipeline

This repository contains end-to-end pipelines for analyzing **real fish** and **robot fish** video data. The two pipelines use **separate Conda environments** and should be installed and run independently.

This README provides a **repo-wide entry point** summarizing installation and execution. Each subdirectory contains more detailed READMEs for advanced usage and configuration.

---

## Installation

> ⚠️ Important: There are **two separate Conda environments**:
>
> * One for **real fish** pipelines
> * One for **robot fish (YOLO-based)** pipelines

---

### Real Fish Environment

Make sure you are running in a Linux or Windows WSL environment

1. **Create or update the Conda environment**
```bash
conda create -n betta python=3.10
conda activate betta
```
2. **Install dependencies**
```bash
pip install -e .
```

---

### Robot Fish Environment (YOLO Pipelines)

All robot fish pipelines (detection, cropping, classification) use a shared YOLO environment.

1. **Create or update the Conda environment**

   ```bash
   conda env update --file environment_yolo.yml
   conda activate yolo-env
   ```

   * If the environment does not exist, it will be created
   * If it exists, it will be updated to match the YAML file

No editable install step is required for robot fish pipelines.

---

## Real Fish Pipeline

To run pose estimation, just run following commands after installation

### Pose Estimation and Feature Extraction

## Comamnd-line Interface
```bash
betta
```
## To run inference
```bash
betta pose_predict
```
**Input**

Directory containing .mp4 video files

**Output**

Directory containing pose prediction CSV files

**Command** 
```bash
betta train --video-folder <VIDEO_FOLDER> --output-folder <OUTPUT_FOLDER> --cfg-file <CONFIG_YAML> --ckpt-file <CHECKPOINT_CKPT>
```
**Example**
```bash
betta train --video-folder ./videos --output-folder ./pose_outputs --cfg-file ./configs/betta.yaml --ckpt-file ./checkpoints/betta.ckpt
```
## To run feature generation
betta feature-generation
**Input**

Directory containing:

Lightning Pose CSV files (.csv)

OR DeepLabCut HDF5 files (.h5)

**Output**

Directory containing generated feature CSV files

**Command**
```bash
betta feature-generation --input <POSE_OUTPUT_FOLDER> --output <FEATURE_OUTPUT_FOLDER>
```
**Example**
```bash
betta feature-generation --input ./pose_outputs --output ./action_features
```

### LITAction (Aggregate Model)

> *To be completed.*

---

## Robot Fish Pipeline

> Reminder: Activate the **YOLO Conda environment** before running any robot fish pipeline.

```bash
conda activate yolo-env
```

---

### Detection and Cropping Pipeline

This pipeline detects a single fish per frame, smooths trajectories, removes outliers, and produces cropped videos centered on the fish.

```bash
python robot_fish/yolo_detection_cropping/detect_and_crop.py \
  --video path/to/video.mp4 \
  --video_name experiment_01 \
  --model path/to/detector.pt \
  --output path/to/output
```

Outputs include:

* Annotated detection videos
* Cropped videos (`*_cropped.mp4`)
* CSV files with detections, smoothed trajectories, and outliers
* Trajectory plots

See `robot_fish/yolo_detection_cropping/README.md` for advanced parameters.

---

### YOLO-Based Classification Pipeline

This pipeline performs **frame-level behavior classification** on cropped videos produced by the detection pipeline.

#### Single Video

```bash
python robot_fish/yolo_classification/run_yolo_classification.py \
  --video path/to/experiment_01_cropped.mp4 \
  --model path/to/classifier.pt \
  --output_dir path/to/output \
  --device cuda
```

#### Multiple Videos

```bash
python robot_fish/yolo_classification/run_yolo_classification.py \
  --video_dir path/to/yolo_detection_cropping_outputs \
  --model path/to/classifier.pt \
  --output_dir path/to/output \
  --device cuda
```

Optional manual labels can be provided for evaluation using `--label_dir`.

Outputs include:

* Per-frame prediction CSVs
* Per-video class summaries
* Confusion matrices (when labels are provided)

See `robot_fish/yolo_classification/README.md` for full details.

---

## Outputs and Version Control

* All pipelines write outputs to **user-specified directories**
* Outputs may include large videos and CSV files
* Output directories should **not** be tracked by git

Each pipeline exposes `--output` or `--output_dir` arguments to support flexible deployment on local machines or clusters.

---

## Notes for Cluster Deployment

* Ensure the correct Conda environment is activated in batch jobs
* GPU pipelines default to CUDA when available
* Paths should point to shared filesystems when running on clusters

---

For detailed documentation, refer to the READMEs inside individual pipeline directories.
