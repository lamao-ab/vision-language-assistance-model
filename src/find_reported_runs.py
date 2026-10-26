"""
find_reported_runs.py
=====================
Scans a folder tree for training runs (each run folder holds training_history.json
next to final_adapter/) and identifies the six runs behind the reported tables:
LoRA and QLoRA, seeds 7 / 42 / 123, rank 8, learning rate 2e-4, max length 512,
3 epochs.

Other runs share some of these settings and are told apart as follows:
  * rank sweep (r = 4, 16, 32)        -> excluded by rank
  * earlier QLoRA runs at lr 2e-5      -> excluded by learning rate
  * mixing-ratio ablation runs         -> same config but FEWER training examples,
                                          so a lower global_step (the reported runs
                                          use all captions and have the highest)

If several runs match one (method, seed), the script prints them side by side and
marks the group AMBIGUOUS instead of guessing.

Usage
-----
  python src/find_reported_runs.py --root drive/MyDrive/VLMs/outputs
"""
import argparse
import csv
import datetime
import json
import os

SEEDS = {7, 42, 123}
KNOWN_BEST_LOSS = {("LoRA", 123): 1.6655, ("QLoRA", 123): 1.6723}   # from the paper, seed 123


def method_of(cfg):
    return "QLoRA" if "qlora" in str(cfg.get("method", "")).lower() else "LoRA"


def load_run(hist_path):
    h = json.load(open(hist_path))
    c = h.get("config", {})
    folder = os.path.dirname(hist_path)
    final = os.path.join(folder, "final_adapter")
    return {
        "folder": folder,
        "final_adapter": final if os.path.isdir(final) else None,
        "method": method_of(c),
        "seed": c.get("seed"),
        "rank": c.get("rank", c.get("lora_rank")),
        "alpha": c.get("alpha", c.get("lora_alpha")),
        "lr": c.get("learning_rate"),
        "max_length": c.get("max_length"),
        "epochs": c.get("num_epochs"),
        "global_step": h.get("global_step"),
        "best_loss": h.get("best_metric"),
        "modified": datetime.datetime.fromtimestamp(os.path.getmtime(hist_path)).strftime("%Y-%m-%d %H:%M"),
    }


def matches_reported(r):
    return (r["rank"] == 8 and r["alpha"] == 16 and r["lr"] is not None
            and abs(r["lr"] - 2e-4) < 1e-12 and r["max_length"] == 512
            and r["epochs"] == 3 and r["seed"] in SEEDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", default="results/all_runs.csv")
    args = ap.parse_args()

    runs = []
    for dirpath, _, files in os.walk(args.root):
        if "training_history.json" in files:
            try:
                runs.append(load_run(os.path.join(dirpath, "training_history.json")))
            except Exception as e:
                print("skipped (unreadable):", dirpath, e)

    print(f"found {len(runs)} runs under {args.root}\n")
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()) if runs else ["folder"])
        w.writeheader()
        w.writerows(runs)

    cand = [r for r in runs if matches_reported(r)]
    print(f"{len(cand)} runs match rank 8 / alpha 16 / lr 2e-4 / max_length 512 / 3 epochs / seed in {sorted(SEEDS)}\n")

    chosen, problems = [], []
    for method in ("LoRA", "QLoRA"):
        for seed in sorted(SEEDS):
            g = sorted([r for r in cand if r["method"] == method and r["seed"] == seed],
                       key=lambda r: -(r["global_step"] or 0))
            tag = f"{method} seed {seed}"
            if not g:
                problems.append(f"{tag}: NO matching run found")
                print(f"  {tag:<16} -> NOT FOUND")
                continue
            if len(g) > 1:
                top, nxt = g[0], g[1]
                print(f"  {tag:<16} -> {len(g)} candidates (AMBIGUOUS unless step counts differ clearly)")
                for r in g:
                    print(f"       step {r['global_step']}  best_loss {r['best_loss']}  {r['modified']}  {r['folder']}")
                if (top["global_step"] or 0) == (nxt["global_step"] or 0):
                    problems.append(f"{tag}: candidates tie on global_step, resolve by hand")
                    continue
            r = g[0]
            known = KNOWN_BEST_LOSS.get((method, seed))
            note = ""
            if known is not None and r["best_loss"] is not None:
                note = "  [best loss matches paper]" if abs(r["best_loss"] - known) < 5e-4 else \
                       f"  [!! best loss {r['best_loss']:.4f} differs from paper value {known}]"
            print(f"  {tag:<16} -> {r['folder']}  step {r['global_step']}  best_loss {r['best_loss']}{note}")
            if r["final_adapter"]:
                chosen.append(r["final_adapter"])
            else:
                problems.append(f"{tag}: no final_adapter folder inside {r['folder']}")

    print("\n" + "=" * 70)
    if problems:
        print("NEEDS ATTENTION:")
        for p in problems:
            print("  -", p)
    if len(chosen) == 6 and not problems:
        print("All six runs identified. Next command:\n")
    elif chosen:
        print("Identified so far (fix the items above, then rerun):\n")
    if chosen:
        print("python src/audit_adapters.py --runs \\")
        for c in chosen:
            print(f"    {c} \\")
        print("    --out results/adapter_audit.json")
    print(f"\nfull table of every run saved to {args.csv}")


if __name__ == "__main__":
    main()
