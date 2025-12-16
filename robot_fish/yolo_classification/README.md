# YOLO Classification Pipeline

This pipeline performs frame-level behavior classification using a YOLO classification model.
It is designed to run on videos produced by the YOLO detection and cropping pipeline.

The input videos are cropped videos centered on the fish, typically named:

<video_name>_cropped.mp4

---

## Pipeline Overview

This pipeline:

1. Takes cropped videos as input
2. Runs a YOLO classification model on each frame
3. Produces per-frame predictions and class probabilities
4. Optionally evaluates predictions against manually annotated labels
5. Saves all results to a structured output directory

This pipeline does NOT perform detection or cropping.
It assumes the input videos already contain a single fish centered in the frame.

---

## Expected Input

This pipeline is intended to be run on the output of the YOLO detection and cropping pipeline.

If the detection pipeline was run as:
```
python detect_and_crop.py \
  --video path/to/video.mp4 \
  --video_name experiment_01 \
  --model detector.pt \
  --output outputs/yolo_detection_cropping
```
Then the classification pipeline expects as input:

outputs/yolo_detection_cropping/experiment_01/videos/experiment_01_cropped.mp4

---

## Environment

This pipeline expects the YOLO Conda environment defined in the repository root.

From the repository root:
```
conda activate yolo-env
```
---

## Basic Usage (Single Video)

To classify a single cropped video:
```
python robot_fish/yolo_classification/run_yolo_classification.py \
  --video outputs/yolo_detection_cropping/experiment_01/videos/experiment_01_cropped.mp4 \
  --model /path/to/classifier_model.pt \
  --output_dir outputs/yolo_classification \
  --device cuda
```
---

## Running on Multiple Videos

To classify all cropped videos produced by the detection pipeline:
```
python robot_fish/yolo_classification/run_yolo_classification.py \
  --video_dir outputs/yolo_detection_cropping \
  --model /path/to/classifier_model.pt \
  --output_dir outputs/yolo_classification \
  --device cuda
```
The script will recursively search for all files matching:

*_cropped.mp4

---

## Using Ground Truth Labels (Optional)

Manually annotated labels are optional and are **not included** in this repository.

If you want to evaluate classification performance, you must point the pipeline to the directory
where the label CSV files are stored.

Expected label naming convention:

manual_scoring_<video_name>.csv

Example directory structure:
```
/path/to/labels/
└── manual_scoring_3558_robot.csv
```
Run with labels:
```
python robot_fish/yolo_classification/run_yolo_classification.py \
  --video outputs/yolo_detection_cropping/3558_robot/videos/3558_robot_cropped.mp4 \
  --label_dir /path/to/labels \
  --model /path/to/classifier_model.pt \
  --output_dir outputs/yolo_classification \
  --device cuda
```
When labels are provided, the pipeline will compute:
- Per-video class rates
- Global classification report
- Confusion matrix

---

## Command-line Arguments

--video  
Path to a single cropped video file (*_cropped.mp4).

--video_dir  
Directory containing cropped videos. The script will search recursively.

--model  
Path to the YOLO classification model (.pt).

--output_dir  
Base directory where outputs will be written.

--device  
Inference device: cuda or cpu.

--label_dir  
Optional directory containing manual labels.

--max_frames  
Optional limit on the number of frames per video (useful for debugging).

--overwrite  
Recompute predictions even if cached CSVs already exist.

---

## Outputs

All outputs are written to the directory specified via the `--output_dir` argument.

The pipeline does not enforce a fixed output location. You are free to choose any
directory structure that fits your workflow.

Typical outputs include:
- Per-frame prediction CSVs
- Optional evaluation summaries (when labels are provided)
- Run metadata for reproducibility

Because outputs may include large files (especially CSVs and videos),
they are not intended to be tracked by git.

---

## Notes

- Output files are not tracked by git.
- Predictions are cached by default; rerunning the pipeline will reuse existing CSVs unless --overwrite is set.
- The pipeline assumes exactly one fish per frame.
- GPU acceleration is used automatically when available.
