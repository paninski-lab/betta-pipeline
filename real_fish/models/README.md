## Models (packaging folder)

This folder is the **packaging location** for the final trained models to be delivered.

It contains minimal artifacts (e.g., `config.yaml` + a `.ckpt` checkpoint) copied from training runs.

The full training outputs (TensorBoard logs, extra checkpoints, predictions, etc.) remain in the
full sweep artifacts directory:

- `hyperparameter_sweep/models/`
