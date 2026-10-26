"""
build_caption_review.py — sample and blind captions for qualitative review
===========================================================================
Produces a blinded scoring sheet for comparing base vs. LoRA captions on the
same VizWiz-Caps validation images.

Design:
  * n = 50 images, stratified:
      - 30 random (typical cases)
      - 10 where the adapted caption is short (<=8 words; harder images)
      - 10 whose captions mention a text-bearing object (label, box, bottle,
        screen, book, ...), so the "missed safety-relevant detail" axis has
        cases to score
  * For each image the two captions are shuffled and labelled A / B, so the
    scorer does not know which model produced which.
  * Outputs:
      caption_review_SHEET.csv  -> fill this in while viewing each image
      caption_review_KEY.csv    -> model identity per row (do NOT open until done)

Run:  python3 build_caption_review.py
"""
import json, csv, re, random

SEED        = 20260819          # fixed for reproducibility
N_RANDOM    = 30
N_SHORT     = 10
N_TEXT      = 10
SHORT_MAX_W = 8

BASE_FILE = "vizwiz_caption_val_predictions_base_generic.json"
LORA_FILE = "vizwiz_caption_val_predictions_lora_seed123_custom.json"

TEXT_CUES = re.compile(
    r"\b(label|package|packag|box|bottle|screen|book|card|sign|medicat|pill|can|jar|"
    r"bag|menu|receipt|paper|monitor|phone|remote|instruction|barcode|tag|cover|"
    r"magazine|newspaper|document)\w*", re.I)


def load(path):
    return {x["image_id"]: x["caption"].strip() for x in json.load(open(path))}


def main():
    rng = random.Random(SEED)
    base = load(BASE_FILE)
    lora = load(LORA_FILE)
    ids = sorted(set(base) & set(lora))

    short_pool = [i for i in ids if len(lora[i].split()) <= SHORT_MAX_W]
    text_pool = [i for i in ids if TEXT_CUES.search(base[i] + " " + lora[i])]

    chosen, used = [], set()

    def take(pool, k, stratum):
        picked = 0
        for i in rng.sample(pool, len(pool)):
            if picked == k:
                break
            if i in used:
                continue
            used.add(i)
            chosen.append((i, stratum))
            picked += 1

    take(short_pool, N_SHORT, "short")
    take(text_pool, N_TEXT, "text")
    take(ids, N_RANDOM, "random")

    rng.shuffle(chosen)

    sheet_rows, key_rows = [], []
    for n, (img_id, stratum) in enumerate(chosen, 1):
        pair = [("base", base[img_id]), ("lora", lora[img_id])]
        rng.shuffle(pair)                      # blind: A/B order randomised
        (m_a, cap_a), (m_b, cap_b) = pair

        sheet_rows.append({
            "row": n,
            "image_file": f"VizWiz_val_{img_id:08d}.jpg",
            "image_id": img_id,
            "stratum": stratum,
            "caption_A": cap_a,
            "A_hallucination": "",       # 1 = asserts something not in the image, else 0
            "A_missed_safety_detail": "",# 1 = omits legible text/identifier/hazard; NA if none present
            "A_relevant": "",            # 1 = answers "what am I looking at?", else 0
            "caption_B": cap_b,
            "B_hallucination": "",
            "B_missed_safety_detail": "",
            "B_relevant": "",
            "notes": "",
        })
        key_rows.append({"row": n, "image_id": img_id,
                         "stratum": stratum, "A_model": m_a, "B_model": m_b})

    with open("caption_review_SHEET.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet_rows[0].keys()))
        w.writeheader(); w.writerows(sheet_rows)

    with open("caption_review_KEY.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader(); w.writerows(key_rows)

    from collections import Counter
    print(f"Wrote caption_review_SHEET.csv ({len(sheet_rows)} rows) and caption_review_KEY.csv")
    print("strata:", dict(Counter(s for _, s in chosen)))
    print("\nSCORING GUIDE")
    print("  hallucination        : 1 if the caption asserts an object, attribute, or text")
    print("                         that is NOT present in the image; else 0")
    print("  missed_safety_detail : 1 if the image contains legible text, an identifier,")
    print("                         or a hazard that the caption omits; NA if none present")
    print("  relevant             : 1 if the caption would answer a blind user's implicit")
    print("                         'what am I looking at?'; else 0")
    print("\nDo NOT open caption_review_KEY.csv until scoring is complete.")


if __name__ == "__main__":
    main()
