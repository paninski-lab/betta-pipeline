## Inference (Lightning-Action · Real Fish)

This folder provides a **simple CLI** to run inference with the trained Lightning-Action models from the Week 14 sweep.

It supports:
- **Subtype models** on their own subtype datasets
- A **single chosen aggregate model** (selected by mean F1 across fish types) evaluated on each subtype dataset
- Running inference on **many videos at once** via either a split file or an explicit session list

### Setup

- Run commands from the **repository root** (the folder that contains `Week 14/`).
- The script auto-detects the repo root by locating `Week 14/hyperparameter_sweep/`.

### Run subtype model inference (best model for that subtype)

```bash
py -3 "Week 14/deliverables/inference/run_lightning_action_inference.py" ^
  --model_kind subtype ^
  --target_subtype fighting ^
  --variant LP_with_cal_contour ^
  --split test
```

### Run the single aggregate model on a subtype dataset

```bash
py -3 "Week 14/deliverables/inference/run_lightning_action_inference.py" ^
  --model_kind aggregate_single ^
  --target_subtype ornamental ^
  --variant LP_with_cal_contour ^
  --split test
```

### Run inference on an explicit list of many sessions

```bash
py -3 "Week 14/deliverables/inference/run_lightning_action_inference.py" ^
  --model_kind aggregate_single ^
  --target_subtype wild ^
  --variant LP_with_cal_contour ^
  --session_ids 1385.1 2.7.1R 4.5.1R 1381.2 1384.1
```

### Run inference on a different dataset folder

Use `--data_path` to point to a folder that contains an input directory (default: `features/`).

```bash
py -3 "Week 14/deliverables/inference/run_lightning_action_inference.py" ^
  --model_kind subtype ^
  --target_subtype fighting ^
  --data_path "D:/some_new_dataset" ^
  --input_dir features ^
  --session_ids videoA videoB videoC
```

### Notes

- If you omit `--hparam_key`, the script selects:
  - subtype model: best key from `hyperparameter_sweep/results/<subtype>_results.json`
  - aggregate model: global best key from `hyperparameter_sweep/results/aggregate_results.json` via mean across fish types
- Ornamental aliases like `2.7.1oR` / `3.5.1oR` are normalized automatically.


