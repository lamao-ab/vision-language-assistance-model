#!/usr/bin/env python3
"""
measure_a100_table.py - fills the "Cloud (A100)" column of the edge-versus-cloud table
======================================================================================
Measures the three cells that can be measured on the GPU:

  * Inference latency   VQA (30 new tokens) and Caption (64 new tokens), mean +- SD per profile
  * Power draw          one instantaneous NVML reading taken right after each timed run
  * Static footprint    torch.cuda.memory_allocated() after the model is loaded

Network latency, end-to-end variance, connectivity, privacy and cost are properties of the
deployment context, not of the GPU. They are NOT measured here and must be given in the
paper as stated assumptions or as a parameterized scenario.

Protocol (identical to the run recorded in a100_table14_output.txt)
--------------------------------------------------------------------
  * model        : the merged 4-bit NF4 QLoRA checkpoint, loaded with from_pretrained and no
                   explicit dtype or quantization argument (the checkpoint carries its own
                   bitsandbytes configuration)
  * input        : ONE fixed image (--image); prompts have no leading <image> token, the
                   processor inserts it and prints a benign warning
  * decoding     : greedy (do_sample=False), batch size 1, max_new_tokens 30 / 64
  * per profile  : preprocessing once (not timed), one untimed warm-up generation, then
                   --runs timed generations (default 30)
  * latency      : torch.cuda.synchronize() on both sides of model.generate() only
  * power        : pynvml nvmlDeviceGetPowerUsage() once, immediately after each timed run
                   (card power only; the host is not included), then a 0.5 s pause
  * clocks       : not locked (shared Colab runtime)

Limits that must be stated in the paper: the power values are single instantaneous
driver-averaged readings, not integrals; they cannot be multiplied by latency to give energy.

Run on the GPU machine (Colab: confirm the GPU first with
`!nvidia-smi --query-gpu=name --format=csv`):

    pip install nvidia-ml-py
    python measure_a100_table.py --image /path/to/image.jpg [--out_json a100_raw.json]
"""
import argparse
import hashlib
import json
import statistics as st
import time

DEFAULT_MODEL = "lamao-ab/paligemma-blind-assist-qlora-merged-v1"

PROFILES = {
    "VQA (30 tok)":     {"prompt": "Assist a blind person: what is in front of me?", "max_new_tokens": 30},
    "Caption (64 tok)": {"prompt": "Describe this scene for a blind person.",         "max_new_tokens": 64},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="the single fixed image used for every run")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=30, help="timed runs per profile")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_json", default=None, help="optional: save every raw latency and power value")
    args = ap.parse_args()

    import torch
    import pynvml
    from PIL import Image
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
    print(f"GPU: {name}")
    print(f"Power limit: {power_limit_w:.1f} W\n")

    print("Loading model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, device_map={"": f"cuda:{args.gpu}"}).eval()
    processor = PaliGemmaProcessor.from_pretrained(args.model)
    torch.cuda.synchronize()

    footprint_gb = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"Static footprint (memory_allocated): {footprint_gb:.2f} GB\n")

    image = Image.open(args.image).convert("RGB")

    results, raw = {}, {}
    for label, cfg in PROFILES.items():
        print(f"--- {label} ---")
        inputs = processor(text=cfg["prompt"], images=image, return_tensors="pt").to(model.device)

        with torch.inference_mode():                       # warm-up, untimed
            model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
        torch.cuda.synchronize()

        lat, power = [], []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            lat.append(t1 - t0)
            power.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)   # one reading per run, right after
            time.sleep(0.5)

        results[label] = {"lat_mean": st.mean(lat), "lat_std": st.pstdev(lat),
                          "lat_min": min(lat), "lat_max": max(lat), "power_mean": st.mean(power)}
        raw[label] = {"latency_s": lat, "power_W": power}
        print(f"  latency  : {st.mean(lat):.3f} ± {st.pstdev(lat):.3f} s  [{min(lat):.3f}-{max(lat):.3f}]")
        print(f"  power    : {st.mean(power):.1f} W (sampled)")
        s = sorted(lat)
        print(f"  (extra)  : median {st.median(lat):.3f} s | p90 {s[int(0.9 * (len(s) - 1))]:.3f} s\n")

    print("=" * 60)
    print("TABLE CELLS")
    print("=" * 60)
    lo = min(r["lat_mean"] for r in results.values())
    hi = max(r["lat_mean"] for r in results.values())
    pw = [r["power_mean"] for r in results.values()]
    print(f"Inference Latency : {lo:.2f}--{hi:.2f} s")
    print(f"Power Draw (mean across profiles) : {st.mean(pw):.1f} W")
    print(f"Static Footprint : {footprint_gb:.2f} GB")
    print(f"(extra) Power draw range across profiles : {min(pw):.1f}--{max(pw):.1f} W  <- preferred for the paper")

    if args.out_json:
        import transformers
        json.dump({
            "gpu": name, "power_limit_W": power_limit_w, "model": args.model,
            "image_sha256": hashlib.sha256(open(args.image, "rb").read()).hexdigest(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "runs_per_profile": args.runs, "static_footprint_gb": footprint_gb,
            "summary": results, "raw": raw,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, open(args.out_json, "w"), indent=2)
        print(f"\nraw values saved -> {args.out_json}")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
