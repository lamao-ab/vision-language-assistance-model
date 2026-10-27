# Caption review: design and scoring rubric

Blinded comparison of the base model (generic prompt) and the adapted LoRA model (seed 123, accessibility prompt) on VizWiz-Captions validation images. The criteria below are the scoring guide printed by `src/build_caption_review.py`.

## Design

- 100 images in two batches of 50, stratified: 60 random (typical cases), 20 where the adapted caption is short (8 words or fewer), 20 whose captions mention a text-bearing object (label, package, box, bottle, screen, book, card, sign, medication, receipt and similar), so that the safety-detail criterion has cases to score.
- For each image the two captions are shuffled and labelled A and B. The model behind A and B is recorded in `caption_review_KEY.csv` and was not opened until scoring was complete.
- One rater, an author, blind to model identity. No second rater was available.

## Criteria (scored per caption)

| Column | Score |
|---|---|
| `hallucination` | 1 if the caption asserts an object, attribute or text that is NOT present in the image; otherwise 0 |
| `missed_safety_detail` | 1 if the image contains legible text, an identifier or a hazard that the caption omits; otherwise 0; NA if no such content is present in the image (58 of the 100 images were rated) |
| `relevant` | 1 if the caption would answer a blind user's implicit question "what am I looking at?"; otherwise 0 |

## Files

- `caption_review_SHEET_100.csv`: the sheet as rated by the author (100 pairs).
- `caption_review_SHEET_100_blank.csv`: the same 100 pairs with all ratings and notes removed, for independent rating.
- `caption_review_KEY_100.csv`: model identity for every A and B (open only after rating).
- `caption_review_stats.json`: single-rater statistics (see `src/caption_review_stats.py`).
