# Changelog

## v2.0 — revised submission (October 2026)

The audited version is **v1.0 (commit `4f4d378`)**. Nothing that produced a reported score was modified: the scorers `src/score_vizwiz_vqa.py` and `src/score_captions.py`, the evaluation script `src/evaluate_vizwiz.py`, the training scripts `src/train_lora.py` and `src/train_qlora.py`, and `data/prepare_dataset.py` are byte-identical to their v1.0 versions (checked with `git diff 4f4d378 HEAD`). Until this release the history since `4f4d378` consisted of additions only.

### Added

**Analysis scripts (`src/`)**
- `inventory_trainable.py`, `audit_adapters.py`: reconstruct the trainable-parameter inventory (modules, parameters, tensors, dtypes) and audit the adapters of the six reported runs.
- `find_reported_runs.py`: locates the six reported runs and their best validation losses.
- `selective_prediction_analysis.py`: retained-set composition by confidence threshold, informative accepted set, cross-fitted thresholds, comparison of two prediction sets of the same model.
- `make_overlap_spec.py`, `audit_overlap.py`, `overlap_exposure_analysis.py`: image-overlap audit across the training, validation and evaluation partitions of both tasks (SHA-256 identity, optional difference hash), exposure categories and sensitivity analyses.
- `diagnose_caption_csv.py`: checks how image names in the released VizWiz-Captions metadata CSV relate to the released splits.
- `check_completeness.py`: asserts that a prediction file covers every eligible item (the scorers are unchanged).
- `score_captions_baseline.py`: caption scoring used for the baseline comparison.
- `paired_comparisons.py`: paired VizWiz-VQA comparisons (gains with bootstrap intervals, prompt and adaptation decomposition, adapter against exported model).
- `caption_review_stats.py`, `build_caption_review.py`, `analyze_caption_review.py`: blinded caption review (sampling, unblinding, single-rater statistics with Wilson intervals and exact McNemar tests).
- `deploy_paligemma_to_hf.py`: the export procedure (merge the adapter in bf16, quantize to 4-bit NF4, push to the Hub).

**Deployment (`deployment/`)**
- `analyze_energy_v3.py` (exact-window integration of the power log), `measure_a100_table.py` (A100 reference measurement).
- `records/`: raw console output, power log, marker file, per-run latencies, A100 output and `SHA256SUMS.txt`.
- Final `measure_memory.py` and `model_breakdown.py` used for the reported memory block.

**Results (`results/`)**
- `selective/`, `overlap/`, `adapters/`, `runs/`, `completeness_report_vqa.json`, `data_preparation_log.txt`, `caption_review/` (blinded review sheet, key and rubric).

**Documentation**
- README: evaluation representations, reproduction table, updated results.

### Changed
- `README.md`: results restated. Added the base model under the accessibility prompt, the gain over the base at its better prompt, development-set labelling of the captioning results, exported-model results, and the restated deployment metrics (memory 2.14 GB static and 4.10 of 7.44 GB used; energy 16.62 J per caption query over the generation window; battery life marked illustrative; A100 comparison limited to power).
- `requirements.txt`: removed an outdated note stating that caption metrics came from a non-official scorer. The reported values are reproduced by the official COCO pipeline (`pycocoevalcap` 1.2 with PTBTokenizer).
- `deployment/requirements.txt`: header states the measured JetPack, PyTorch and CUDA versions and no longer refers to files that are not in the repository.

### Moved
- `notebooks ` (folder name with a trailing space) to `notebooks/`.
- `configs/training_config.yaml` to `archive/training_config_stale.yaml`. No script reads it, and its values (learning rate 2e-5, dropout 0.05, maximum length 256) differ from the defaults of the training scripts (2e-4, 0.03, 512).

### Corrections to the documentation (code and reported scores unchanged)
- Trainable parameters: the reconstructed inventory gives 11,298,816 trainable parameters (0.385%): 126 decoder modules (9,805,824 parameters) and 81 vision-encoder `q/k/v` modules (1,492,992 parameters), 207 modules and 414 adapter tensors, in float32. The earlier description of the adapted modules and parameter count is corrected.
- Deployment: the energy per query is restated from an exact-window integration (the earlier analysis summed only the samples inside each window); the A100 static footprint is 2.14 GB; the energy-reduction claim against the A100 is withdrawn because the measurement boundaries differ.

Tag `v1.0` is unchanged.
