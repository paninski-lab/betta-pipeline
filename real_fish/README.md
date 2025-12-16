## Real Fish · Lightning-Action (packaged deliverables)

This folder packages the **final code deliverables** for the Real Fish / Lightning-Action pipeline:

- **Inference pipeline + README**
- **Training pipeline for subtype-specific models + configs**
- **Training pipeline for aggregate models + configs**

These scripts assume you have the **Week 14 sweep artifacts** available as a folder named:

- `hyperparameter_sweep/`

in the **repository root** (the folder that contains `real_fish/`).

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
  - Reference config YAMLs for **best subtype models** and the **single best aggregate model**
- `models/`
  - Minimal model artifacts copied for packaging

### Notes

- These scripts require `lightning_action` to be importable in your Python environment.
- Run commands from the **repository root** (e.g., the `betta-pipeline/` folder).

### Large artifacts (Drive)

The full sweep artifacts (especially the full model directory) are stored on Google Drive:

- `hyperparameter_sweep/` (full folder): `https://drive.google.com/drive/u/0/folders/1VhN2DE2rhxwyhF8vSUFdVIhfXA2xV_0c`
