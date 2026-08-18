"""
bench_latency.py  — CORRECTED latency benchmark
=================================================
Fixes vs. original latency_breakdown.py:
  * Stage 1 no longer times an artificial time.sleep(0.8); it measures the
    REAL cost of acquiring a fresh frame (buffer flush + one read).
  * TTS is split into SYNTHESIS (espeak -> wav) and PLAYBACK (paplay), so the
    fast 28 ms synth is never confused with the seconds of audio playback.
  * Both task profiles are measured: VQA (30 new tokens) and CAPTION (64).
    -> backs the two separate "Inference Latency" rows in the paper.
  * Preprocess stage is CUDA-synced so the H2D copy isn't mis-attributed.
  * Throughput is computed from a clearly DEFINED end-to-end, not from a
    number that secretly contained the 3.4 s sleep.
  * Run conditions (nvpmodel mode) are recorded into the output.

Run ON THE JETSON, camera connected.
"""
import cv2, torch, time, subprocess, os
import numpy as np
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

MODEL_ID      = "lamao-ab/paligemma-blind-assist-qlora-merged-v1"
NUM_RUNS      = 30
WARMUP_RUNS   = 5
BUFFER_FLUSH  = 5                 # grabs to drop stale V4L2 frames (NOT a sleep)
TEMP_WAV      = "/tmp/bench_tts.wav"
COUNT_PLAYBACK = True             # measure paplay audio playback in E2E

TASKS = {
    "VQA":     {"prompt": "<image>Assist a blind person: what is in front of me?", "max_new_tokens": 30},
    "Caption": {"prompt": "<image>Describe this scene for a blind person.",        "max_new_tokens": 64},
}

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"

def record_conditions():
    print("="*60)
    print("RUN CONDITIONS")
    print("="*60)
    print("nvpmodel        :", sh("nvpmodel -q | tr '\\n' ' '"))
    print("model_id        :", MODEL_ID)
    print("torch / cuda    :", torch.__version__, "/", torch.version.cuda)
    print("="*60 + "\n")

# ── Load model ────────────────────────────────────────────────────────────
record_conditions()
print("Loading model...")
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
).eval()

# ── Init camera (BUFFERSIZE=1 keeps frames fresh without a long sleep) ──────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 )
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception: pass
for _ in range(30): cap.grab()          # one-time pipeline prime (not timed)
time.sleep(1.0)

def acquire_frame():
    """Flush stale buffers then read ONE fresh frame. Real, no artificial sleep."""
    for _ in range(BUFFER_FLUSH): cap.grab()
    ret, frame = cap.read()
    if not ret: return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ── Warmup ──────────────────────────────────────────────────────────────────
print("Warming up...")
img = acquire_frame()
for _ in range(WARMUP_RUNS):
    inp = processor(text=TASKS["Caption"]["prompt"], images=img, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()
print("Warmup done.\n")

def stats(a): a = np.array(a); return a.mean(), a.std(), a.min(), a.max()

def run_task(name, cfg):
    t_cam, t_pre, t_inf, t_dec, t_synth, t_play = [], [], [], [], [], []
    for i in range(NUM_RUNS):
        # 1) camera acquisition (REAL fresh-frame cost)
        t0 = time.perf_counter()
        image = acquire_frame()
        t_cam.append((time.perf_counter() - t0) * 1000)
        if image is None:
            print("  camera read failed, skipping run"); continue

        # 2) preprocess / tokenize  (synced)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        inputs = processor(text=cfg["prompt"], images=image, return_tensors="pt").to("cuda")
        torch.cuda.synchronize(); t_pre.append((time.perf_counter() - t0) * 1000)

        # 3) inference  (synced both sides)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
        torch.cuda.synchronize(); t_inf.append((time.perf_counter() - t0) * 1000)

        # 4) decode
        t0 = time.perf_counter()
        text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        t_dec.append((time.perf_counter() - t0) * 1000)

        # 5) TTS synthesis (espeak -> wav)
        t0 = time.perf_counter()
        subprocess.run(['espeak', '-s', '165', '-v', 'en+m3', '-w', TEMP_WAV, text or "."],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t_synth.append((time.perf_counter() - t0) * 1000)

        # 6) TTS playback (paplay) — measured separately
        if COUNT_PLAYBACK:
            t0 = time.perf_counter()
            subprocess.run(['paplay', TEMP_WAV], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t_play.append((time.perf_counter() - t0) * 1000)
        if os.path.exists(TEMP_WAV): os.remove(TEMP_WAV)

        print(f"  [{name}] run {i+1:02d}/{NUM_RUNS}  inf={t_inf[-1]:.0f}ms  cam={t_cam[-1]:.0f}ms")
    return dict(cam=t_cam, pre=t_pre, inf=t_inf, dec=t_dec, synth=t_synth, play=t_play)

results = {name: run_task(name, cfg) for name, cfg in TASKS.items()}
cap.release()

# ── Report ────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("LATENCY BREAKDOWN  (ms, mean ± std)")
print("="*72)
for name, r in results.items():
    print(f"\n--- {name} (max_new_tokens={TASKS[name]['max_new_tokens']}) ---")
    rows = [("1. Camera acquisition (flush+read)", r["cam"]),
            ("2. Preprocess / tokenize",           r["pre"]),
            ("3. Model inference",                  r["inf"]),
            ("4. Token decode",                     r["dec"]),
            ("5. TTS synthesis (espeak)",           r["synth"])]
    if COUNT_PLAYBACK: rows.append(("6. TTS playback (paplay)", r["play"]))
    for label, arr in rows:
        m, s, mn, mx = stats(arr)
        print(f"  {label:<36} {m:>7.1f} ± {s:<6.1f}  [{mn:.0f}-{mx:.0f}]")

    compute = np.array(r["pre"]) + np.array(r["inf"]) + np.array(r["dec"])
    e2e_sys = compute + np.array(r["cam"]) + np.array(r["synth"])
    print(f"  {'COMPUTE (pre+inf+dec)':<36} {compute.mean():>7.1f} ± {compute.std():<6.1f}")
    print(f"  {'END-TO-END system (+cam +synth)':<36} {e2e_sys.mean():>7.1f} ± {e2e_sys.std():<6.1f}")
    print(f"  -> Inference latency to cite : {np.mean(r['inf'])/1000:.2f} ± {np.std(r['inf'])/1000:.2f} s")
    print(f"  -> Throughput (system E2E)   : {1000/e2e_sys.mean():.3f} queries/s")
    if COUNT_PLAYBACK:
        e2e_user = e2e_sys + np.array(r["play"])
        print(f"  -> User-perceived E2E        : {e2e_user.mean()/1000:.2f} s  (incl. audio playback)")
print("\nNote: throughput is 1 / END-TO-END (system). It now matches the latency rows.")