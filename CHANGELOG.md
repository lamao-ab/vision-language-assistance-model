# Changelog

## v2.0 — revised submission (November 2026)

The audited version is **v1.0 (commit `4f4d378`)**. The VQA scorer `src/score_vizwiz_vqa.py`, the training scripts `src/train_lora.py` and `src/train_qlora.py` and `data/prepare_dataset.py` are byte-identical to their v1.0 versions (checked with `git diff 4f4d378 HEAD`). `src/score_captions.py` was extended, and `src/evaluate_vizwiz.py` and `src/evaluate_baselines.py` gained a caption test mode (see Changed); their VQA and caption-validation code and their decoding are unchanged. `src/evaluate_benchmark.py` and `src/predict.py` differ from v1.0 only in a corrected example Hub ID in their docstrings and help texts (see Corrections). Apart from the files listed under Changed, Moved and Corrections, every change since v1.0 is an addition.

### Added

**Analysis scripts (`src/`)**
- `inventory_trainable.py`, `audit_adapters.py`: reconstruct the trainable-parameter inventory (modules, parameters, tensors, dtypes) and audit the adapters of the six reported runs.
- `find_reported_runs.py`: locates the six reported runs and their best validation losses.
- `selective_prediction_analysis.py`: retained-set composition by confidence threshold, informative accepted set, cross-fitted thresholds, comparison of two prediction sets of the same model.
- `make_overlap_spec.py`, `audit_overlap.py`, `overlap_exposure_analysis.py`, `overlap_dedup_test_captions.py`, `overlap_label_audit.py`: image-overlap audit (and a comparison of the questions, answers and captions of the shared photographs) across the training, validation and test partitions of both tasks (SHA-256 identity, optional difference hash), exposure categories and sensitivity analyses on the VQA and caption test images.
- `diagnose_caption_csv.py`: checks how image names in the released VizWiz-Captions metadata CSV relate to the released splits.
- `check_completeness.py`: asserts that a prediction file covers every eligible item.
- `caption_test_checks.py`: validity checks of the realigned test references (neighbour-shift test, position check, duplicate sensitivity).
- `caption_paired_intervals.py`: paired VizWiz-Captions comparisons (CIDEr-D gains and QLoRA against LoRA with bootstrap intervals, exported model against its adapter).
- `build_evaluation_manifest.py`: the evaluation manifest linking every reported run to its checkpoint, representation, precision, prompt, decoding, scorer and prediction file (SHA-256).
- `paired_comparisons.py`: paired VizWiz-VQA comparisons (gains with bootstrap intervals, prompt and adaptation decomposition, adapter against exported model).
- `caption_review_stats.py`, `build_caption_review.py`, `analyze_caption_review.py`: blinded caption review (sampling, unblinding, single-rater statistics with Wilson intervals and exact McNemar tests).
- `deploy_paligemma_to_hf.py`: the export procedure (merge the adapter in bf16, quantize to 4-bit NF4, push to the Hub); it uploads the reviewed model cards kept in `hub_cards/`.

**Deployment (`deployment/`)**
- `analyze_energy_v3.py` (exact-window integration of the power log), `measure_a100_table.py` (A100 reference measurement).
- `records/`: raw console output, power log, marker file, per-run latencies, A100 output and `SHA256SUMS.txt`.
- Final `measure_memory.py` and `model_breakdown.py` used for the reported memory block.

**Results (`results/`)**
- `selective/`, `overlap/` (six partitions: VQA and captions, each with train, validation and test), `adapters/`, `runs/`, `completeness_report_vqa.json`, `completeness_report_caps.json`, `data_preparation_log.txt`, `caption_review/` (rated sheet, blank sheet for independent rating, key, rubric, single-rater statistics).
- `caption_test/` (the VizWiz-Captions test references aligned to the test images, their description and a worksheet for checking the alignment), `evaluation_manifest.csv`, `paired_comparisons.json`, `caption_paired_intervals.json`.
- `predictions/` (prediction files and per-run score files of the reported runs, the zero-shot baselines and the exported models) and `predictions/benchmarks/` (VQAv2 test submissions, COCO caption predictions and the returned server outputs and score files).
- `hub_cards/` (the reviewed Hugging Face model cards).

**Documentation**
- README: evaluation representations, reproduction table, updated results.

### Changed
- `src/score_captions.py`: scores VizWiz-Captions validation and test references (the realigned `test_captions_7974.json`) and COCO with the same metrics and reference filtering as in v1.0. Non-ASCII characters are folded to ASCII before scoring; compared with no folding, the test CIDEr-D of the PaliGemma runs changes by 0.01 to 0.06 and the validation CIDEr-D by at most 0.01. The script scores all models, including the zero-shot baselines.
- `src/evaluate_vizwiz.py`, `src/evaluate_baselines.py`: caption test mode (`--caps_split test`, with `--vizwiz_test_json` and `--vizwiz_test_images`). The VQA path, the caption-validation path (including its output file names) and the decoding are unchanged.
- `README.md`: results restated. Added the base model under the accessibility prompt, the gain over the base at its better prompt, captioning results on the test captions, exported-model results, and the restated deployment metrics (memory 2.14 GB static and 4.10 of 7.44 GB used; energy 16.62 J per caption query over the generation window; battery life marked illustrative; A100 comparison limited to power).
- `requirements.txt`: removed an outdated note stating that caption metrics came from a non-official scorer. The reported values are reproduced by the official COCO pipeline (`pycocoevalcap` 1.2 with PTBTokenizer).
- `deployment/requirements.txt`: header states the measured JetPack, PyTorch and CUDA versions and no longer refers to files that are not in the repository.

### Moved
- `notebooks ` (folder name with a trailing space) to `notebooks/`.
- `configs/training_config.yaml` to `archive/training_config_stale.yaml`. No script reads it, and its values (learning rate 2e-5, warmup ratio 0.05, maximum length 256) differ from the settings of the training scripts (learning rate 2e-4, warmup ratio 0.03, maximum length 512).

### Corrections to the documentation (code and reported scores unchanged)
- Hub model ID: the docstrings and help texts of `src/evaluate_vizwiz.py`, `src/evaluate_benchmark.py` and `src/predict.py` named `lamao-ab/paligemma-blind-assist-jetson-ready`, which is not among the public models. They now name `lamao-ab/paligemma-blind-assist-qlora-merged-v1` (7 lines, no code changed). The README links the correct GitHub repository for issues, and states the Hub repositories and pinned revisions: LoRA `lamao-ab/paligemma-blind-assist-lora-merged-v1` at `0545d03`, QLoRA merged `lamao-ab/paligemma-blind-assist-qlora-merged-v1` at `0a66cea`.
- Trainable parameters: the reconstructed inventory gives 11,298,816 trainable parameters (0.385%): 126 decoder modules (9,805,824 parameters) and 81 vision-encoder `q/k/v` modules (1,492,992 parameters), 207 modules and 414 adapter tensors, in float32. The earlier description of the adapted modules and parameter count is corrected.
- Deployment: the energy per query is restated from an exact-window integration (the earlier analysis summed only the samples inside each window); the A100 static footprint is 2.14 GB; the energy-reduction claim against the A100 is withdrawn because the measurement boundaries differ.

Tag `v1.0` is unchanged.
