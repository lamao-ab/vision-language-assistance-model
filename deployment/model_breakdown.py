"""
model_breakdown.py  — CORRECTED model memory breakdown
======================================================
Fixes vs. original:
  * Activation + KV cache: original read memory_allocated() AFTER generate()
    returned (intermediates already freed -> bogus 9 MB). Now uses
    reset_peak_memory_stats() + max_memory_allocated() to get the true PEAK.
  * 4-bit weight accounting: bitsandbytes Params4bit already stores PACKED
    bytes (2 values per uint8), so nelement*element_size IS the byte count.
    The original then did `nelement // 2`, halving it a second time. Fixed.
  * Adds a dtype histogram so you can SEE what is 4-bit vs fp16 (the big "other"
    bucket is the ~257k-token embedding table + SigLIP vision tower — not noise).
  * Prints whether the model actually loaded quantized, so the paper can state
    it correctly instead of "fp16 + 4-bit".

Run ON THE JETSON.
"""
import torch
from collections import defaultdict
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image

MODEL_ID = "lamao-ab/paligemma-blind-assist-qlora-merged-v1"

print("Loading model...")
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
).eval()

# ── 0. Is it actually quantized? ──────────────────────────────────────────
qcfg = getattr(model.config, "quantization_config", None)
print("\nquantization_config:", qcfg if qcfg else "NONE (running in fp16)")

# ── 1. Parameter accounting ───────────────────────────────────────────────
base_bytes = quant_bytes = lora_bytes = fp16_bytes = 0
dtype_hist = defaultdict(lambda: [0, 0])   # dtype -> [count, bytes]

for name, p in model.named_parameters():
    n = p.nelement()
    is_4bit = hasattr(p, "quant_state") or p.__class__.__name__ == "Params4bit"
    if is_4bit:
        b = n * p.element_size()           # already packed bytes — do NOT // 2
        base_bytes += b
        qs = getattr(p, "quant_state", None)
        if qs is not None:
            try:
                quant_bytes += sum(t.nelement()*t.element_size()
                                   for t in qs if isinstance(t, torch.Tensor))
            except TypeError:
                pass
        dtype_hist["4bit(NF4)"][0] += n; dtype_hist["4bit(NF4)"][1] += b
    else:
        b = n * p.element_size()
        if "lora" in name.lower(): lora_bytes += b
        else:                      fp16_bytes += b
        dtype_hist[str(p.dtype)][0] += n; dtype_hist[str(p.dtype)][1] += b

for name, buf in model.named_buffers():
    b = buf.nelement()*buf.element_size()
    fp16_bytes += b
    dtype_hist[f"buffer/{buf.dtype}"][0] += buf.nelement()
    dtype_hist[f"buffer/{buf.dtype}"][1] += b

mb = lambda x: x/1024**2
gb = lambda x: x/1024**3

print("\n" + "="*60); print("DTYPE HISTOGRAM"); print("="*60)
print(f"{'dtype':<22}{'params':>16}{'MB':>12}")
for d,(c,b) in sorted(dtype_hist.items(), key=lambda kv:-kv[1][1]):
    print(f"{d:<22}{c:>16,}{mb(b):>12.1f}")

print("\n" + "="*60); print("MODEL MEMORY BREAKDOWN"); print("="*60)
print(f"Base weights (4-bit NF4)   : {mb(base_bytes):>9.1f} MB  ({gb(base_bytes):.3f} GB)")
print(f"Quantization constants     : {mb(quant_bytes):>9.1f} MB")
print(f"LoRA adapters              : {mb(lora_bytes):>9.1f} MB")
print(f"Other non-quant (bf16 embed+vision) : {mb(fp16_bytes):>9.1f} MB  ({gb(fp16_bytes):.3f} GB)")
static = base_bytes + quant_bytes + lora_bytes + fp16_bytes
print("-"*60)
print(f"Static model footprint     : {gb(static):.3f} GB")

# ── 2. Activation + KV cache — TRUE PEAK during a real forward pass ────────
print("\nMeasuring peak activation + KV cache (corrected method)...")
dummy = Image.new("RGB", (320, 240), (128, 128, 128))
inputs = processor(text="<image>Describe this scene for a blind person.",
                   images=dummy, return_tensors="pt").to("cuda")
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_allocated()
with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()
peak_activation = max(peak - before, 0)

print("\n" + "="*60); print("FINAL SUMMARY FOR PAPER"); print("="*60)
print(f"Base model weights (4-bit)    : {gb(base_bytes):.3f} GB")
print(f"Quantization constants        : {mb(quant_bytes):.1f} MB")
print(f"LoRA adapters                 : {mb(lora_bytes):.1f} MB")
print(f"Other non-quant (bf16 embed+vis) : {gb(fp16_bytes):.3f} GB")
print(f"Static footprint              : {gb(static):.3f} GB")
print(f"Activation + KV cache (PEAK)  : {gb(peak_activation):.3f} GB  ({mb(peak_activation):.0f} MB, measured)")
print(f"Peak CUDA reserved            : {gb(torch.cuda.max_memory_reserved()):.3f} GB")