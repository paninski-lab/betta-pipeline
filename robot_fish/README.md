# Robot Fish Pipelines

This repository contains end-to-end pipelines for detecting, tracking, and classifying fish behavior in video data.

---

## Repository Structure
```
betta-pipeline/
├── pipelines/
│   ├── yolo_detection_cropping/
│   │   ├── median_gaussian.py
│   │   └── README.md
│   └── (future pipelines)
├── outputs/               # Generated outputs (ignored by git)
├── environment_yolo.yml   # Conda environment for YOLO pipelines
├── requirements.txt       # Python dependencies
└── README.md
```
---

## Environment Setup (YOLO Pipelines)

All YOLO-based pipelines (detection, cropping, classification) are expected to run in the same Conda environment.

### Create or update the environment

From the repository root:
```
conda env update --file environment_yolo.yml
conda activate yolo-env
```
Notes:
- If the environment does not exist, this command will create it.
- If the environment already exists, it will be updated to match the configuration file.
- Make sure Conda is installed and available in your system before running these commands.

---

## Outputs and Version Control

All pipelines write their outputs (videos, CSVs, plots) to a user-specified directory.

Each pipeline exposes an `--output` or `--output_dir` argument that allows you to choose
where results are saved. There is no fixed output directory enforced by this repository.

Because outputs can be very large (especially videos), output directories should not be
tracked by version control and should remain outside the repository or be explicitly ignored.

Output locations are entirely flexible and can be adapted to local machines, shared
filesystems, or cloud-based environments.

---

## Available Pipelines

### YOLO Detection and Cropping

This pipeline:
- Runs YOLO detection on video frames
- Tracks a single fish per frame
- Smooths trajectories and detects outliers
- Produces cropped videos centered on the fish

See:
pipelines/yolo_detection_cropping/README.md

---
