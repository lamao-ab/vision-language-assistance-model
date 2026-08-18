"""
measure_memory.py — system memory measurement (footprint read directly, not back-solved)
"""
import subprocess, time

def free_snapshot(n=5, pause=0.3):
    tot=used=avail=0
    for _ in range(n):
        out = subprocess.check_output("free -m", shell=True).decode()
        for line in out.splitlines():
            if line.startswith("Mem:"):
                p = line.split()
                tot += int(p[1]); used += int(p[2]); avail += int(p[6])
        time.sleep(pause)
    return tot/n/1024, used/n/1024, avail/n/1024

def line(): print("-"*55)

print("="*55); print("STAGE 1 — OS BASELINE (model + app NOT running)"); print("="*55)
tot, used_os, avail_os = free_snapshot()
print(f"Total     : {tot:.2f} GB")
print(f"Used (OS) : {used_os:.2f} GB")
print(f"Available : {avail_os:.2f} GB")

input("\n[Enter] to load the model and measure the full footprint...")

print("\n" + "="*55); print("STAGE 2 — MODEL LOADED"); print("="*55)
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
MODEL_ID = "lamao-ab/paligemma-blind-assist-qlora-merged-v1"
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
).eval()
torch.cuda.synchronize(); time.sleep(2)

tot2, used_model, avail_model = free_snapshot()
cuda_alloc = torch.cuda.memory_allocated()/1024**3
cuda_resv  = torch.cuda.memory_reserved()/1024**3

print(f"Total     : {tot2:.2f} GB")
print(f"Used      : {used_model:.2f} GB")
print(f"Available : {avail_model:.2f} GB")

print("\n" + "="*55); print("SUMMARY"); print("="*55)
print(f"Total memory             : {tot2:.2f} GB")
print(f"Model footprint (CUDA)   : {cuda_alloc:.2f} GB   <- direct, reproducible")
print(f"CUDA reserved            : {cuda_resv:.2f} GB")
line()
print(f"System used after load   : {used_model:.2f} GB   <- device utilization (model + libs + OS)")
print(f"OS baseline (context)    : {used_os:.2f} GB")
print(f"Available after load     : {avail_model:.2f} GB")