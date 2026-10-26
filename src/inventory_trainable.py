"""
inventory_trainable.py
======================
Builds the LoRA or QLoRA model EXACTLY as src/train_lora.py / src/train_qlora.py
do (same loading call, same k-bit preparation, same LoraConfig), then records
every parameter tensor: name, shape, dtype, trainable status. No training is run.

Purpose: tie the method description and parameter accounting in the paper to the
configuration that produced the reported results (reviewer request), including
which components carry adapters and the dtypes of trainable / frozen tensors.

Outputs
-------
  <out>.json   summary: totals, trainable parameters per component, adapted
               modules per component, dtype histograms, software versions
  <out>.csv    one row per tensor (name, shape, dtype, requires_grad, numel)

Usage
-----
  python src/inventory_trainable.py --mode lora  --out results/inventory_lora
  python src/inventory_trainable.py --mode qlora --out results/inventory_qlora

  # what a decoder-only configuration would give (NOT what the released scripts do)
  python src/inventory_trainable.py --mode lora --scope decoder --out results/inventory_lora_decoder

Note: for 4-bit weights `numel_stored` counts packed uint8 storage (two 4-bit
values per byte), so frozen-weight counts under QLoRA are storage counts.
"""
import argparse
import collections
import csv
import json
import os
import platform

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# regex that restricts the same suffixes to the Gemma language model only
DECODER_ONLY = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def component(name: str) -> str:
    if "language_model" in name or name.endswith("lm_head.weight") or "lm_head" in name:
        return "decoder"
    if "vision_tower" in name:
        return "vision"
    if "multi_modal_projector" in name:
        return "projector"
    return "other"


def summarize(records):
    """records: iterable of dicts(name, shape, dtype, requires_grad, numel, is_quant)."""
    records = list(records)
    is_adapter = lambda n: ".lora_A." in n or ".lora_B." in n

    trainable = [r for r in records if r["requires_grad"]]
    out = {
        "n_tensors": len(records),
        "n_trainable_tensors": len(trainable),
        "trainable_params": sum(r["numel"] for r in trainable),
        "total_params_stored": sum(r["numel"] for r in records),
    }
    out["trainable_pct_of_stored"] = round(100 * out["trainable_params"] / out["total_params_stored"], 4)

    by_comp = collections.Counter()
    for r in trainable:
        by_comp[component(r["name"])] += r["numel"]
    out["trainable_params_by_component"] = dict(by_comp)

    mods = collections.defaultdict(set)
    for r in records:
        if ".lora_A." in r["name"]:
            mods[component(r["name"])].add(r["name"].split(".lora_A.")[0])
    out["adapted_modules_by_component"] = {k: len(v) for k, v in mods.items()}
    out["adapted_module_suffixes_by_component"] = {
        k: dict(collections.Counter(m.rsplit(".", 1)[-1] for m in v)) for k, v in mods.items()}

    out["trainable_non_adapter_tensors"] = [r["name"] for r in trainable if not is_adapter(r["name"])]
    out["adapter_dtypes"] = dict(collections.Counter(r["dtype"] for r in records if is_adapter(r["name"])))
    frozen = [r for r in records if not r["requires_grad"]]
    h = collections.defaultdict(lambda: {"tensors": 0, "numel_stored": 0})
    for r in frozen:
        h[r["dtype"]]["tensors"] += 1
        h[r["dtype"]]["numel_stored"] += r["numel"]
    out["frozen_dtypes"] = dict(h)
    out["frozen_quantized_tensors"] = sum(1 for r in frozen if r.get("is_quant"))
    return out


def build_model(mode, scope, model_id, rank):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import PaliGemmaForConditionalGeneration

    if mode == "lora":                                   # as src/train_lora.py
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, revision="bfloat16")
        model.to("cuda")
    else:                                                # as src/train_qlora.py
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=q, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    cfg = LoraConfig(
        r=rank, lora_alpha=rank * 2,
        target_modules=TARGETS if scope == "all" else DECODER_ONLY,
        task_type=TaskType.CAUSAL_LM, lora_dropout=0.05)
    return get_peft_model(model, cfg)


def versions():
    import importlib
    v = {"python": platform.python_version()}
    for m in ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]:
        try:
            v[m] = importlib.import_module(m).__version__
        except Exception:
            v[m] = None
    try:
        import torch
        v["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lora", "qlora"], required=True)
    ap.add_argument("--scope", choices=["all", "decoder"], default="all",
                    help="all = suffix list on the whole model (as released); "
                         "decoder = restricted to the Gemma language model")
    ap.add_argument("--model_id", default="google/paligemma-3b-mix-224")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--out", required=True, help="output path prefix (no extension)")
    args = ap.parse_args()

    model = build_model(args.mode, args.scope, args.model_id, args.rank)
    records = [dict(name=n, shape=list(p.shape), dtype=str(p.dtype).replace("torch.", ""),
                    requires_grad=p.requires_grad, numel=p.numel(),
                    is_quant=p.__class__.__name__ == "Params4bit")
               for n, p in model.named_parameters()]

    summary = summarize(records)
    summary.update(mode=args.mode, scope=args.scope, rank=args.rank,
                   model_id=args.model_id, versions=versions())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "shape", "dtype", "requires_grad", "numel", "is_quant"])
        w.writeheader()
        for r in records:
            w.writerow({**r, "shape": "x".join(map(str, r["shape"]))})

    print(json.dumps({k: summary[k] for k in [
        "trainable_params", "trainable_pct_of_stored", "trainable_params_by_component",
        "adapted_modules_by_component", "adapter_dtypes", "trainable_non_adapter_tensors"]}, indent=2))
    model.print_trainable_parameters()
    print(f"\nsaved -> {args.out}.json / .csv")


if __name__ == "__main__":
    main()
