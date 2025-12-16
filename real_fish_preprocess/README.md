# Real Fish Preprocess

**Real Fish Preprocess** is a command-line toolkit for extracting behavioral features from betta fish videos.  
It provides a two-stage workflow:

1. **Pose inference** using pretrained **Lightning Pose** models  
2. **Feature generation** from pose outputs for downstream **Lightning Action** models

The pipeline is designed for **inference-only usage**, works without NVIDIA DALI, and is compatible with Windows and Linux.

---

## Installation

pip install -e .

## Comamnd-line Interface

betta

## To run inference
betta train

**Input**

Directory containing .mp4 video files

**Output**

Directory containing pose prediction CSV files

**Command** 
betta train --video-folder <VIDEO_FOLDER> --output-folder <OUTPUT_FOLDER> --cfg-file <CONFIG_YAML> --ckpt-file <CHECKPOINT_CKPT>

**Example**
betta train --video-folder ./videos --output-folder ./pose_outputs --cfg-file ./configs/betta.yaml --ckpt-file ./checkpoints/betta.ckpt

## To run feature generation
betta feature-generation
**Input**

Directory containing:

Lightning Pose CSV files (.csv)

OR DeepLabCut HDF5 files (.h5)

**Output**

Directory containing generated feature CSV files

**Command**

betta feature-generation --input <POSE_OUTPUT_FOLDER> --output <FEATURE_OUTPUT_FOLDER>

**Example**

betta feature-generation --input ./pose_outputs --output ./action_features