# Real Fish Preprocess

**Real Fish Preprocess** is a command-line toolkit for extracting behavioral features from betta fish videos.  
It provides a two-stage workflow:

1. **Pose inference** using pretrained **Lightning Pose** models  
2. **Feature generation** from pose outputs for downstream **Lightning Action** models

The pipeline is designed for **inference-only usage**, works without NVIDIA DALI, and is compatible with Windows and Linux.

---

## Installation

To run inside the lightning-pose conda environment on the cluster:

```bash
module load anaconda/3-2023.09
conda activate lightning-pose

# Ensure the env's bin has priority
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

pip install -e .
```

## Command-line Interface

betta

## To run inference
> ⚠️ **GPU requirement**
>
> The provided Lightning Pose checkpoints are saved on CUDA.
> To run pose inference with `betta train`, use a **GPU node** on the cluster.
> Running on a CPU-only node will fail when loading the checkpoint.

```bash
betta pose_predict
```

**Input**

Directory containing .mp4 video files

**Output**

Directory containing pose prediction CSV files

**Command** 
```bash
betta pose-predict --video-folder <VIDEO_FOLDER> --output-folder <OUTPUT_FOLDER> --cfg-file <CONFIG_YAML> --ckpt-file <CHECKPOINT_CKPT>
```

**Example**
```bash
betta pose-predict --video-folder ./videos --output-folder ./pose_outputs --cfg-file ./configs/betta.yaml --ckpt-file ./checkpoints/betta.ckpt
```

## To run feature generation

```bash
betta feature-generation
```

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
