## Reference Configs (Week 14 · Hyperparameter Sweep)

These YAML files are **reference configs** for training the selected models:

- Best subtype-specific model per fish type (chosen by max test F1 within subtype)
- Single aggregate model (chosen by mean test F1 across fish types)

Two flavors are provided:
- `*_train.yaml`: trained on **train split only**
- `*_all.yaml`: trained on **all split** (train+test+val)

Model keys (as of current sweep results):
- fighting best: `nh64_ly3_lg8_lr5e-04`
- hybrid best: `nh32_ly3_lg8_lr5e-04`
- ornamental best: `nh32_ly3_lg4_lr1e-04`
- wild best: `nh64_ly2_lg8_lr1e-04`
- aggregate single best: `nh64_ly3_lg8_lr5e-04` (mean across fish types ≈ 0.6594)

### Path conventions

- `data.data_path` is written as a **relative path** from the repo root, e.g. `Week 14/hyperparameter_sweep/...`.
- Run training commands from the repo root (the folder that contains `Week 14/`).

You can train directly from these with:

```bash
py -3 -c "from lightning_action.api import Model; import yaml; cfg=yaml.safe_load(open(r'PATH_TO_CONFIG')); Model.from_config(cfg).train(output_dir=r'OUTPUT_DIR')"
```


