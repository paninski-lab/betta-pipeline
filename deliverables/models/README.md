## Models (packaging folder)

This folder is the **packaging location** for the final trained models to be delivered.

During training runs launched from `Week 14/week14_hyperparameter_sweep.ipynb`, the notebook copies the minimal artifacts here:

- `aggregate_all_data/<hparam_key>/`
  - `config.yaml`
  - `final_model.ckpt` (or a `.ckpt` checkpoint if `final_model.ckpt` is not produced)
  - `MODEL_POINTER.json` (paths + metadata)

The full training outputs (TensorBoard logs, extra checkpoints, etc.) remain under:

- `Week 14/hyperparameter_sweep/models/`


