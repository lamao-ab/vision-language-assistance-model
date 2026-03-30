import torch
import time
import numpy as np
import psutil
import os
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# CONFIGURATION
MODEL_ID = "lamao-ab/paligemma-blind-assist-jetson-ready"
NUM_RUNS = 20 
VQA_PROMPT = "<image>Assist a blind person: What color is the shirt?"
CAP_PROMPT = "<image>Describe this scene for a blind person."

def get_current_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

print("="*50)
print("PHASE 1: SYSTEM BASELINE")
print("="*50)
print("Ensure tegrastats is running in your other terminal.")
input(">>> Note the 'VDD_IN' value now (System Base). Press [Enter] to LOAD MODEL...")

# --- 1. MODEL LOADING ---
print("\n⌛ Loading model weights...")
start_load = time.time()
processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-mix-224")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"},
    low_cpu_mem_usage=True
).eval()
load_time = time.time() - start_load
peak_mem = get_current_mem()

print("\n" + "="*50)
print("PHASE 2: MODEL RESIDENT (IDLE)")
print("="*50)
print(f"Model Loading Time: {load_time:.1f} seconds")
print(f"Memory Footprint:   {peak_mem:.2f} GB")
input(">>> Note the new 'VDD_IN' value (Model Resident). Press [Enter] to START INFERENCE...")

# --- 2. WARMUP ---
print("\n🔥 Warming up GPU kernels...")
dummy_img = Image.new('RGB', (224, 224), color='white')
dummy_input = processor(text=CAP_PROMPT, images=dummy_img, return_tensors="pt").to("cuda")
with torch.inference_mode():
    _ = model.generate(**dummy_input, max_new_tokens=10)

# --- 3. BENCHMARKING FUNCTIONS ---
def run_benchmark(prompt, max_tokens, label):
    print(f"⌛ Benchmarking {label} (Active Phase)...")
    latencies = []
    inputs = processor(text=prompt, images=dummy_img, return_tensors="pt").to("cuda")
    
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize() 
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    
    return np.mean(latencies), np.std(latencies)

# --- 4. EXECUTION ---
vqa_mean, vqa_std = run_benchmark(VQA_PROMPT, 30, "VQA")
cap_mean, cap_std = run_benchmark(CAP_PROMPT, 64, "Captioning")

# 5. FINAL REPORT
throughput = 1.0 / cap_mean

print("\n" + "="*50)
print("📝 FINAL RESULTS FOR TABLE 6")
print("="*50)
print(f"1. Model Load Time:      {load_time:.1f} s")
print(f"2. Peak Memory Usage:    {peak_mem:.2f} GB")
print(f"3. Latency (VQA):        {vqa_mean:.2f} ± {vqa_std:.2f} s")
print(f"4. Latency (Caption):    {cap_mean:.2f} ± {cap_std:.2f} s")
print(f"5. Throughput:           {throughput:.2f} images/s")
print("="*50)
print(">>> Check tegrastats logs for the VDD_IN spikes during Phase 3.")
