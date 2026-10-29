# archive

Files that are superseded but kept for provenance.

## training_config_stale.yaml

The training configuration file of v1.0, moved here unchanged. **No script reads it**: neither `src/train_lora.py` nor `src/train_qlora.py` loads a YAML file. It records an earlier QLoRA setting (learning rate 2e-5, warmup ratio 0.05, maximum length 256).

The reported models were trained with the defaults of the training scripts: learning rate 2e-4, warmup ratio 0.03 (set in the script), maximum length 512, 3 epochs. The LoRA dropout is 0.05 in both the YAML and the scripts. The settings of each reported run are stored in its own `training_history.json` and are summarised in `results/runs/reported_runs.csv`.
