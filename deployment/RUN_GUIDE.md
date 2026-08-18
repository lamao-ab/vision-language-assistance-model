# Re-running the Jetson measurements — procedure

All five scripts run **on the Jetson Orin Nano 8 GB**. Copy them to the device
(e.g. `~/bench/`). Do the steps in order; the whole thing is ~15 minutes.

## 0. Fix the run conditions ONCE (reproducibility)

The old energy log ran with dynamic clocks, so the numbers weren't repeatable.
Lock the power mode and clocks before measuring, and report which mode you used:

```bash
sudo nvpmodel -m 0        # 0 = MAXN (max perf). Use the mode you want to report.
sudo jetson_clocks         # lock clocks so DVFS doesn't move under you
nvpmodel -q                # confirm; note this string in the paper
```

Close the GUI / browser / the assistant app so nothing else uses GPU or RAM.

## 1. Memory — system footprint  (`measure_memory.py`)

```bash
python3 measure_memory.py
```

It pauses after the OS-baseline read. With **nothing else running**, press Enter
to load the model. Record from the SUMMARY block:
`Total`, `OS + kernel baseline`, `Model + runtime footprint`, `Available after load`.
These are all measured and sum to total — no residual.

## 2. Memory — model internals  (`model_breakdown.py`)

```bash
python3 model_breakdown.py
```

Note the `quantization_config` line (this is what you state in the paper instead
of "fp16 + 4-bit"), the DTYPE HISTOGRAM, and the **PEAK** activation+KV figure.
The `NvMapMemAlloc... error 12` lines are harmless Jetson allocator warnings.

## 3. Latency  (`bench_latency.py`)

Camera connected:

```bash
python3 bench_latency.py
```

Gives separate VQA and Caption tables. Cite **Inference latency** from the
`-> Inference latency to cite` line, and **Throughput** from
`-> Throughput (system E2E)` — they are now consistent (throughput = 1/E2E).
Set `COUNT_PLAYBACK = True` at the top if you also want user-perceived latency
including audio playback.

## 4. Energy  (`energy_run.py` then `analyze_energy.py`)

```bash
python3 energy_run.py        # ~3 min: idle, load, idle, 30 inferences
python3 analyze_energy.py    # prints P_sys / P_idle / P_active, J/query, battery
```

`energy_run.py` samples tegrastats at 100 ms and marks every window, so
`analyze_energy.py` knows exactly which samples are idle vs. inference. Edit
`BATTERY_WH` and `QUERIES_HR` at the top of `analyze_energy.py` to match the
assumptions you want to report.

---

## What to confirm before you publish the new table

- **Power mode**: pick one (MAXN vs 15 W vs 7 W) and report it. Energy is mode-dependent.
- **Battery pack**: 100 Wh is large for a wearable — keep it only if that's your real target; otherwise set your actual pack.
- **Duty cycle**: 10 queries/hr is light; state it as an explicit assumption.
- **VQA latency**: this now comes from a real 30-token run, not a scaled estimate.

If `tegrastats`, `espeak`, or `paplay` aren't found, install them
(`sudo apt install alsa-utils pulseaudio-utils espeak`); tegrastats ships with JetPack.
