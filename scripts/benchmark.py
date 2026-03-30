"""
benchmark.py
============
Measure GPU latency (ms) and power draw (W) for VQA and captioning inference,
matching the A100 benchmark reported in the notebook.

Usage
-----
python scripts/benchmark.py \
    --model_id lamao-ab/paligemma-blind-assist-jetson-ready \
    --n_runs   20

Expected output (A100 SXM4-40 GB):
  VQA     latency: 257 ± N ms
  Caption latency: 537 ± N ms
"""

import argparse
import statistics
import subprocess
import threading
import time

import torch
from PIL import Image as PILImage
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

DEFAULT_MODEL = "lamao-ab/paligemma-blind-assist-jetson-ready"


# ── Power monitor thread ──────────────────────────────────────────────────────

class PowerMonitor(threading.Thread):
    """Poll nvidia-smi every 100 ms and record power draw in watts."""

    def __init__(self):
        super().__init__(daemon=True)
        self.readings: list[float] = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True,
                )
                self.readings.extend(float(x) for x in out.strip().split("\n") if x)
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self):
        self._stop.set()

    def summary(self) -> dict:
        if not self.readings:
            return {"peak_W": None, "mean_W": None}
        return {
            "peak_W": round(max(self.readings),  1),
            "mean_W": round(statistics.mean(self.readings), 1),
        }


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_id: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    return model, processor


# ── Single inference timing ───────────────────────────────────────────────────

@torch.inference_mode()
def timed_generate(
    model,
    processor,
    image: PILImage.Image,
    prompt: str,
    max_new_tokens: int,
) -> float:
    """Return wall-clock latency in milliseconds."""
    inputs = processor(
        images=[image], text=[prompt], return_tensors="pt"
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Synchronise CUDA before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    model,
    processor,
    n_runs: int,
    max_new_tokens_vqa: int,
    max_new_tokens_cap: int,
) -> dict:
    # Use a synthetic 224×224 white image
    dummy = PILImage.new("RGB", (224, 224), color=(128, 128, 128))

    vqa_prompt = "<image>Assist a blind person: What is in front of me?"
    cap_prompt = "<image>Describe this scene for a blind person."

    print(f"\nWarming up (5 runs) …")
    for _ in range(5):
        timed_generate(model, processor, dummy, vqa_prompt, max_new_tokens_vqa)
        timed_generate(model, processor, dummy, cap_prompt, max_new_tokens_cap)

    monitor = PowerMonitor()
    monitor.start()

    print(f"Benchmarking {n_runs} runs per task …")
    vqa_times, cap_times = [], []

    for i in range(n_runs):
        vqa_times.append(
            timed_generate(model, processor, dummy, vqa_prompt, max_new_tokens_vqa)
        )
        cap_times.append(
            timed_generate(model, processor, dummy, cap_prompt, max_new_tokens_cap)
        )
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{n_runs} …")

    monitor.stop()
    monitor.join(timeout=1.0)

    def stats(times):
        return {
            "mean_ms":  round(statistics.mean(times),   1),
            "std_ms":   round(statistics.stdev(times),  1),
            "min_ms":   round(min(times),                1),
            "max_ms":   round(max(times),                1),
        }

    return {
        "vqa_latency":     stats(vqa_times),
        "caption_latency": stats(cap_times),
        "power":           monitor.summary(),
        "device":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_runs":          n_runs,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PaliGemma GPU latency benchmark")
    parser.add_argument("--model_id",           default=DEFAULT_MODEL)
    parser.add_argument("--n_runs",             type=int, default=20)
    parser.add_argument("--max_new_tokens_vqa", type=int, default=20)
    parser.add_argument("--max_new_tokens_cap", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)

    results = run_benchmark(
        model, processor,
        args.n_runs,
        args.max_new_tokens_vqa,
        args.max_new_tokens_cap,
    )

    print("\n── Benchmark Results ────────────────────────────")
    print(f"  Device : {results['device']}")
    v = results["vqa_latency"]
    c = results["caption_latency"]
    p = results["power"]
    print(f"  VQA     latency : {v['mean_ms']} ± {v['std_ms']} ms  "
          f"[{v['min_ms']} – {v['max_ms']}]")
    print(f"  Caption latency : {c['mean_ms']} ± {c['std_ms']} ms  "
          f"[{c['min_ms']} – {c['max_ms']}]")
    if p["peak_W"] is not None:
        print(f"  Peak power      : {p['peak_W']} W")
        print(f"  Mean power      : {p['mean_W']} W")
    print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
