"""
energy_run.py  — controlled energy capture
===========================================
Replaces the ad-hoc tegrastats log. It:
  1. records nvpmodel mode,
  2. samples tegrastats at 100 ms (5x finer than the old 533 ms log),
  3. marks three windows with exact timestamps:
        IDLE_SYS    - system idle, model NOT loaded   -> P_sys
        IDLE_MODEL  - model loaded but idle           -> P_idle
        INFER_k     - each isolated model inference    -> P_active, energy/query
  4. isolates MODEL inference (one frame grabbed up front, reused; NO camera or
     TTS inside the timed window) so energy/query is the model's, matching the
     inference-latency row.

Outputs: power_log.txt (tegrastats) and markers.csv (window boundaries).
Then run:  python3 analyze_energy.py
Run ON THE JETSON.
"""
import subprocess, time, csv, os, signal
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

MODEL_ID    = "lamao-ab/paligemma-blind-assist-qlora-merged-v1"
NUM_INFER   = 30
IDLE_SECS   = 25            # length of each idle baseline window
MAX_TOKENS  = 64           # Caption profile (heavier -> conservative P_active)
POWER_LOG   = "power_log.txt"
MARKERS     = "markers.csv"
TEGRA_INT   = 100          # ms

def now(): return time.time()

def sh(cmd):
    try: return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "n/a"

print("nvpmodel:", sh("nvpmodel -q | tr '\\n' ' '"))
print("Recommend a FIXED power mode + locked clocks for reproducibility:")
print("   sudo nvpmodel -m 0   &&   sudo jetson_clocks      (then re-run)\n")

# start tegrastats -> log
if os.path.exists(POWER_LOG): os.remove(POWER_LOG)
teg = subprocess.Popen(["tegrastats", "--interval", str(TEGRA_INT), "--logfile", POWER_LOG])
time.sleep(2)

markers = []
def mark(label, t0, t1): markers.append((label, f"{t0:.3f}", f"{t1:.3f}"))

# ── IDLE_SYS : model not loaded ────────────────────────────────────────────
print(f"[IDLE_SYS] idle baseline {IDLE_SECS}s (model not loaded)...")
t0 = now(); time.sleep(IDLE_SECS); mark("IDLE_SYS", t0, now())

# ── load model + grab one frame ────────────────────────────────────────────
print("Loading model...")
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(30): cap.grab()
ret, frame = cap.read()
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret \
        else Image.new("RGB", (640, 480), (128, 128, 128))
cap.release()
inputs = processor(text="<image>Describe this scene for a blind person.",
                   images=image, return_tensors="pt").to("cuda")

# warmup
with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()

# ── IDLE_MODEL : model loaded, idle ────────────────────────────────────────
print(f"[IDLE_MODEL] idle baseline {IDLE_SECS}s (model loaded)...")
t0 = now(); time.sleep(IDLE_SECS); mark("IDLE_MODEL", t0, now())

# ── INFER : isolated model inferences, spaced so each burst is clean ────────
print(f"[INFER] {NUM_INFER} isolated inferences...")
for k in range(NUM_INFER):
    torch.cuda.synchronize(); t0 = now()
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    torch.cuda.synchronize(); t1 = now()
    mark(f"INFER_{k}", t0, t1)
    print(f"  infer {k+1:02d}/{NUM_INFER}  {(t1-t0)*1000:.0f} ms")
    time.sleep(1.0)   # let power fall back so bursts don't merge

time.sleep(2)
teg.send_signal(signal.SIGINT); teg.wait(timeout=5)

with open(MARKERS, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["label", "t_start", "t_end"]); w.writerows(markers)

print(f"\nDone. Wrote {POWER_LOG} and {MARKERS}.")
print("Now run:  python3 analyze_energy.py")
