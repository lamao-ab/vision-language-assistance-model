"""
audit_adapters.py
=================
Reads the saved adapter of each reported run (adapter_model.safetensors +
adapter_config.json) and reports what was actually trained: number of tensors,
parameters per component, dtypes, rank/alpha and the target-module list.

Only the safetensors HEADER is parsed, so no model is loaded and no GPU is
needed; it runs in seconds. This ties the parameter accounting in the paper to
the adapters produced by each reported run, not to a re-built configuration.

Usage
-----
  python src/audit_adapters.py \
      --runs drive/MyDrive/VLMs/outputs/lora_seed7_rank8/final_adapter \
             drive/MyDrive/VLMs/outputs/lora_seed42_rank8/final_adapter \
             drive/MyDrive/VLMs/outputs/lora_seed123_rank8/final_adapter \
             drive/MyDrive/VLMs/outputs/qlora_seed7_rank8/final_adapter \
             drive/MyDrive/VLMs/outputs/qlora_seed42_rank8/final_adapter \
             drive/MyDrive/VLMs/outputs/qlora_seed123_rank8/final_adapter \
      --out results/adapter_audit.json
"""
import argparse
import collections
import json
import os
import struct


def read_header(path):
    """Parse a .safetensors header: 8-byte little-endian length, then JSON."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def component(key):
    if "language_model" in key:
        return "decoder"
    if "vision_tower" in key:
        return "vision"
    return "other"


def audit(run_dir):
    st = os.path.join(run_dir, "adapter_model.safetensors")
    header = read_header(st)
    numel = lambda shape: int(__import__("math").prod(shape)) if shape else 1

    out = {"run": run_dir, "n_tensors": len(header), "params": 0,
           "params_by_component": collections.Counter(),
           "dtypes": collections.Counter(), "modules_by_component": collections.defaultdict(set)}
    for key, meta in header.items():
        n = numel(meta["shape"])
        c = component(key)
        out["params"] += n
        out["params_by_component"][c] += n
        out["dtypes"][meta["dtype"]] += 1
        if ".lora_A." in key:
            out["modules_by_component"][c].add(key.split(".lora_A.")[0])

    out["params_by_component"] = dict(out["params_by_component"])
    out["dtypes"] = dict(out["dtypes"])
    out["modules_by_component"] = {k: len(v) for k, v in out["modules_by_component"].items()}

    cfg_path = os.path.join(run_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        out["config"] = {k: cfg.get(k) for k in
                         ["r", "lora_alpha", "lora_dropout", "target_modules",
                          "base_model_name_or_path", "peft_type", "task_type"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="final_adapter directories")
    ap.add_argument("--out", default="results/adapter_audit.json")
    args = ap.parse_args()

    results = [audit(r) for r in args.runs]
    print(f"{'run':<58}{'tensors':>8}{'params':>13}{'decoder':>11}{'vision':>10}  dtypes")
    for r in results:
        c = r["params_by_component"]
        print(f"{os.path.basename(os.path.dirname(r['run'])):<58}{r['n_tensors']:>8}{r['params']:>13,}"
              f"{c.get('decoder', 0):>11,}{c.get('vision', 0):>10,}  {r['dtypes']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
