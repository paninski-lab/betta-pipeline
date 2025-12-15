## Training (Lightning-Action · Real Fish)

This folder contains **reproducible training scripts** for:
- subtype-specific binary models (flare vs background)
- aggregate binary model (trained on all subtypes)

These scripts are aligned with the Week 14 sweep folder structure:

- Data: `Week 14/hyperparameter_sweep/binary_data/<subtype>/<variant>/`
- Models: `Week 14/hyperparameter_sweep/models/<subtype>/`
- Results: `Week 14/hyperparameter_sweep/results/*.json`

### Setup

- Run commands from the **repository root** (the folder that contains `Week 14/`).
- The scripts auto-detect the repo root by locating `Week 14/hyperparameter_sweep/`.

### Subtype-specific training

Train the best model for a subtype (according to `<subtype>_results.json`):

```bash
py -3 "Week 14/deliverables/training/train_subtype_models.py" ^
  --subtype fighting ^
  --variant LP_with_cal_contour ^
  --use_best_from_results ^
  --split train
```

Train the same best hyperparams but on **all** data (train+test+val):

```bash
py -3 "Week 14/deliverables/training/train_subtype_models.py" ^
  --subtype fighting ^
  --variant LP_with_cal_contour ^
  --use_best_from_results ^
  --split all
```

Write `config.yaml` with a **relative** `data_path` (recommended for portability):

```bash
py -3 "Week 14/deliverables/training/train_subtype_models.py" ^
  --subtype fighting ^
  --variant LP_with_cal_contour ^
  --use_best_from_results ^
  --split train ^
  --relative_paths
```

### Aggregate training

Train **one single aggregate model** selected by **mean F1 across fish types** (evenly weighted):

```bash
py -3 "Week 14/deliverables/training/train_aggregate_models.py" ^
  --variant LP_with_cal_contour ^
  --use_best_global_from_results ^
  --split train
```

Train that single aggregate model on **all** data:

```bash
py -3 "Week 14/deliverables/training/train_aggregate_models.py" ^
  --variant LP_with_cal_contour ^
  --use_best_global_from_results ^
  --split all
```

### Notes

- `--post_inference` can be enabled to run automatic inference over training sessions after training completes.
- Ornamental aliases (`2.7.1oR`, `3.5.1oR`) are automatically normalized for training.


