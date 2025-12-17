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

All real fish pipelines use a shared Conda environment and are installed from this repository via an **editable install**.

1. **Create the Conda environment**

  ```bash
  conda create -n betta-real python=3.10 -y
  conda activate betta-real
  ```

2. **Install this repo (editable)**

Run from the repository root (the folder that contains `pyproject.toml`):

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

> Reminder: Activate the **real fish Conda environment** before running any real fish pipeline.
>
  ```bash
  conda activate betta-real
  ```

### Pose Estimation and Feature Extraction

> *To be completed.*

### LITAction (Aggregate Model)

This repository includes packaged Lightning Action pipelines under `real_fish/`:

- Inference: `real_fish/inference/`
- Training: `real_fish/training/`
- Reference configs: `real_fish/configs/`

#### Inference (CLI)
Run inference with the Lightning Action CLI:

  ```bash
  litaction predict --model-dir path/to/trained_model_dir --data-dir path/to/data --input-dir features --output-dir predictions/
  ```

#### Inference (batch evaluation)
Run inference on many sessions at once using the provided script:

  ```bash
  py -3 "real_fish/inference/run_lightning_action_inference.py" ^
    --model_kind aggregate_single ^
    --target_subtype wild ^
    --variant LP_with_cal_contour ^
    --split test
  ```

See `real_fish/inference/README.md` for additional options (explicit session lists, custom data paths, subtype models).

#### Training
Train subtype-specific models:

  ```bash
  py -3 "real_fish/training/train_subtype_models.py" --help
  ```

Train aggregate models:

  ```bash
  py -3 "real_fish/training/train_aggregate_models.py" --help
  ```

See `real_fish/training/README.md` for detailed training instructions and config conventions.

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
