## Reference Configs (Hyperparameter Sweep)

These YAML files are **reference configs** for training the selected models:

- Best subtype-specific model per fish type (chosen by max test F1 within subtype)
- Single aggregate model (chosen by mean test F1 across fish types)

Two flavors are provided:

- `*_train.yaml`: trained on **train split only**
- `*_all.yaml`: trained on **all split** (train+test+val)

### Path conventions

- `data.data_path` is written as a **relative path** from the repo root, e.g. `hyperparameter_sweep/binary_data/...`.
- Run commands from the repo root (the folder that contains `real_fish/`).

You can train directly from these with:

```bash
py -3 -c "from lightning_action.api import Model; import yaml; cfg=yaml.safe_load(open(r'PATH_TO_CONFIG')); Model.from_config(cfg).train(output_dir=r'OUTPUT_DIR')"
```
