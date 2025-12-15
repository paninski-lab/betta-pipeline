## Deliverables (Real Fish · Lightning-Action)

This folder packages the **final code deliverables** for the Real Fish pipeline (Lightning-Action):

- **Inference pipeline + README**
- **Training pipeline for subtype-specific models + configs**
- **Training pipeline for aggregate models + configs**

Everything is designed to work with the Week 14 hyperparameter sweep layout:

- `Week 14/hyperparameter_sweep/binary_data/`
- `Week 14/hyperparameter_sweep/models/`
- `Week 14/hyperparameter_sweep/results/`

### Contents

- `inference/`
  - `run_lightning_action_inference.py`
  - `README.md`
- `training/`
  - `train_subtype_models.py`
  - `train_aggregate_models.py`
  - `common.py`
  - `README.md`
- `configs/`
  - Reference config YAMLs for **best subtype models** and **single best aggregate model**
- `models/`
  - Minimal model artifacts copied from training runs for packaging

### Notes

- These scripts expect the custom package `lightning_action` to be importable (the project includes it in `lightning-action-main/`).
- Run commands from the repository root (the folder that contains `Week 14/`).
- The scripts auto-detect the repo root by locating `Week 14/hyperparameter_sweep/`.

### Large artifacts (Drive)

The full Week 14 sweep artifacts (especially the full model directory) are stored on Google Drive:

- `Week 14/hyperparameter_sweep/` (full folder): `https://drive.google.com/drive/u/0/folders/1VhN2DE2rhxwyhF8vSUFdVIhfXA2xV_0c`


