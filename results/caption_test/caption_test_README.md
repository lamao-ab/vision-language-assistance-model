# VizWiz-Captions test references (official captions, matched to the images)

`test_captions_7974.json` holds captions for 7,974 of the 8,000 VizWiz-Captions test images, in the same format as the official `train.json` and `val.json`. It exists because the standard JSON annotations withhold the test captions and the evaluation server is no longer maintained; the officially released evaluation-annotation file contains them. The captions come from the project metadata file `VizWiz_captions_project_all_metadata_test.csv`, which stores them under project-internal identifiers (`VizWiz_test_000000020000` to `...027999`) that do not correspond to the image file names (`VizWiz_test_00000000.jpg` to `...07999.jpg`).

## Construction

- The captions are stored sequentially rather than at their identifier positions, so subtracting 20,000 from the identifier misassigns captions after the first gap.
- The correspondence was recovered by aligning the per-image `text_detected` flag in `test.json` with the majority vote (at least 3 of 5) of `TEXT_DETECT` in the metadata file, using a monotone sequence alignment, and by resolving the remaining ambiguous positions through manual inspection of the images.
- 7,974 images are matched, with full agreement of the `text_detected` flag; 9 caption sets (identifiers 20631, 20688, 20767, 21788, 23347, 23424, 23477, 25314, 25547) match no image; test images `VizWiz_test_00007974.jpg` to `VizWiz_test_00007999.jpg` (26) have no captions in the released file.
- All 5 captions per image are kept (39,870 captions), each with `is_rejected` and `is_precanned` flags as in `val.json`. The spell-checked captions (`IMG_CAPTION_SPELLCHECKED`) are used, matching the caption version in `val.json`. `is_rejected` is taken from `IS_REJECTED` (817 captions); `is_precanned` is set only for the exact sentence "Quality issues are too severe to recognize visual content." (4,315 captions). Image ids follow `test.json` (from 31181); annotation ids continue the official numbering after `val.json` (155905 to 195774).

## Checks on the released file

Same format as `val.json`; file-name numbers contiguous 0 to 7973; 5 captions per image; annotation ids contiguous; `is_precanned` equals the exact sentence for all 4,315; image-level `text_detected` equals the majority of the five annotation-level flags on 7,974 of 7,974 images (chance level 0.54); 7,807 images have at least one valid caption (not rejected, not pre-canned); mean words per valid caption 12.08 against 12.04 on validation.

## Validity checks on the references (run with the predictions of the trained models)

- Neighbour-shift test: scoring each caption against the references of the previous and the next image, at most 0.6% of the test images (47 of 7,807) fit a neighbour's references better than their own, against 1.1 to 1.4% on the validation images, whose references are official.
- Position check: no block of 100 consecutive images falls more than 3 standard deviations below the mean; 9.7% of the test images score below 10 against 10.4% on validation.
- Both checks are in `src/caption_test_checks.py` and `results/caption_test_checks.json`.

## Limitations

The captions are the official ones; the matching of caption sets to image files is ours. A mis-placed position in the alignment could shift captions only within a run of identical `text_detected` flags (the longest run is 23 images). Misassigned captions would add noise to every model alike. Scores on this file were not produced by the (deprecated) evaluation server.

## Scorer

`src/score_captions_test.py` applies the same metrics and the same reference filtering as `src/score_captions.py`, after folding curly quotes and accents to ASCII (30 test references and 13 predictions contain non-ASCII characters, which the Java tokenizer can split or merge lines on). On the test predictions it gives CIDEr-D 0.01 to 0.06 higher than `score_captions.py`; on the validation predictions the two differ by at most 0.01.

## Use

```bash
python src/score_captions_test.py --dataset vizwiz --gt results/caption_test/test_captions_7974.json --pred <predictions>.json
```
