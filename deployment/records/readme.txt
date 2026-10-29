DEPLOYMENT RECORDS  (deployment/records/)
=========================================
The records behind the deployment numbers in the paper (Tables 11 to 14).
Jetson Orin Nano 8 GB (15 W mode, clocks locked) and one A100 run in Colab.


1. FILES
--------
deployment_results_console_raw.txt
    RAW console record of the Jetson sessions: memory (2 measurement blocks), latency (3 runs),
    energy analysis printouts. This is the file "Deployment Results.txt", unmodified. It is the evidence.
deployment_results.txt
    Organized companion to the raw file. Every numeric block is a verbatim excerpt with its raw line
    numbers and a "USED FOR" tag; two tables marked DERIVED were computed from the printed values.
power_log.txt
    Raw tegrastats log of the energy capture (VDD_IN rail, about 100 ms, 1,870 samples). Device clock
    stamps are whole seconds, MM-DD-YYYY HH:MM:SS (first line 07-04-2026 18:26:39).
markers.csv
    Window boundaries of that capture, epoch seconds: IDLE_SYS, IDLE_MODEL, INFER_0 ... INFER_29.
latency_runs_new.csv
    Inference and camera time of every query of the three latency runs (180 rows). Columns: run, task,
    query, inference_ms, camera_ms, raw_line (line of the value in the raw file).
energy_summary_table13_pair.json
    Output of deployment/analyze_energy_v3.py on power_log.txt + markers.csv, with input checksums.
a100_table14_output.txt
    Console output of the A100 measurement (cloud column of Table 14), verbatim.
SHA256SUMS.txt
    Checksums of every file in this folder.


2. WHERE EACH NUMBER IN THE PAPER COMES FROM
--------------------------------------------
Table 11  Memory and footprint
    System rows (OS baseline 1.00, used 4.10, free 0.11, buffers/cache 3.23, available 3.09 GB):
        deployment_results.txt section 1.3  = raw lines 556-594 (final measure_memory.py)
    Footprint rows (static 2.135 GB, 4-bit weights 1.115 GB, other non-quantized 0.984 GB,
        quantization constants 36.5 MB, activation + KV cache 0.098 GB, peak reserved 2.254 GB):
        section 1.4 = raw lines 622-660 (final model_breakdown.py)
Table 12  Latency
    Run 3 of three consecutive runs: section 2.3 = raw lines 457-487
        VQA inference 0.93 +- 0.11 s, caption 1.97 +- 0.26 s, system E2E 1.36 / 2.41 s,
        user-perceived 2.37 / 5.88 s, throughput 0.74 / 0.42 q/s
    Runs 1 and 2 (raw lines 235-265, 349-379) are reported as a range, not discarded.
Table 13  Power and energy
    As printed by the fixed analyzer: section 3.2 = raw lines 664-692
        P_sys 4.11, P_idle 4.14, P_active 9.60 W; 15.80 +- 0.41 J total, 8.67 +- 0.22 J net;
        24.0 h at 10 queries/h and 10.4 h continuous (100 Wh pack, illustrative)
    Exact-window re-analysis of the same files: energy_summary_table13_pair.json
        16.62 +- 0.16 J total, 9.02 +- 0.15 J above idle, +2.02 +- 0.08 J in the following second
Table 14  Cloud column (A100)
    a100_table14_output.txt: latency 0.266 +- 0.039 s (VQA, 30 tokens) and 0.572 +- 0.007 s (caption,
    64 tokens); power 76.7 W and 64.7 W (one instantaneous NVML reading after each run); static
    footprint 2.14 GB. Jetson column: Tables 12 and 13.


3. HOW TO VERIFY
----------------
    sha256sum -c SHA256SUMS.txt

    # exact-window energy (should print P_sys 4.11 | P_idle 4.14 | P_active 9.60,
    # total 16.62 +- 0.16 J, net 9.02 +- 0.15 J)
    python deployment/analyze_energy_v3.py --log power_log.txt --markers markers.csv --out check.json

    # the analyzer that produced the printed Table 13 values (run in the folder that holds
    # power_log.txt and markers.csv; prints 15.80 +- 0.41 J and 8.67 +- 0.22 J)
    python deployment/analyze_energy.py

    Latency: the per-query values in latency_runs_new.csv reproduce the printed mean and SD of every
    run, task and stage within rounding (the benchmark prints each query in whole milliseconds; the
    largest difference is 0.13 ms).


4. CONDITIONS OF THE MEASUREMENTS
---------------------------------
Device         Jetson Orin Nano 8 GB, nvpmodel -m 0 (15 W), jetson_clocks
Model          lamao-ab/paligemma-blind-assist-qlora-merged-v1 (4-bit NF4 through the checkpoint config)
Software       JetPack 6.2.1 (L4T R36.4.7), PyTorch 2.10.0, CUDA 12.6 (stated in the paper; torch and CUDA are
               also printed by the benchmark); transformers and bitsandbytes versions were not recorded
Environment    PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True
Latency        bench_latency.py: live camera frames at 640x480, predefined text prompts, 30 queries
               per task per run, 5 warm-up runs; VQA cap 30 new tokens, caption cap 64
Energy         energy_run.py: one fixed frame, caption prompt, 64-token cap, 30 isolated queries
               followed by 1 s rest; 25 s idle windows with and without the model; tegrastats at 100 ms
A100           NVIDIA A100-SXM4-40GB, 400 W power limit, Colab runtime, clocks not locked; merged
               4-bit checkpoint, batch size 1, greedy decoding, one fixed image (file not retained), one warm-up and
               30 timed runs per profile, one NVML power reading after each run


5. KNOWN LIMITS
---------------
- Generated-token counts were not recorded. max_new_tokens is a limit, not the output length.
- Latency (live frames, both tasks) and energy (one fixed frame, caption task) are separate protocols;
  they are reported separately and not combined.
- Energy rests on one capture of 30 queries. Memory and latency are repeated (2 and 3 runs).
- The old analyzer printouts (raw lines 493-543) used whole-second timestamps and are not used.
- The A100 power values are single instantaneous card-power readings, not integrals, and exclude the host.
  They support a power-draw comparison only, not an energy comparison.
- The Jetson clock was not verified against a time server; dates in logs are the device's own clock.
- The installed versions of transformers and bitsandbytes on the Jetson were not recorded at measurement time;
  the minimum versions are listed in deployment/requirements.txt.
- The image used in the A100 measurement was not retained, and generated lengths were not recorded. The latency
  ratio of the two profiles (0.572 / 0.266 s = 2.15) is close to the ratio of the caps (64 / 30 = 2.13), which
  is consistent with generation running to the cap, but this is an inference.


6. SCRIPTS (deployment/)
------------------------
As run:       bench_latency.py, energy_run.py, analyze_energy.py (the fixed version used for Table 13)
Older:        measure_memory.py and model_breakdown.py in this repository are the EARLIER versions
              (raw lines 12-39, 62-92). The final versions that produced Table 11 (raw lines 556-660)
              are on the Jetson and are to be copied into the repository.
Written for the revision:
              analyze_energy_v3.py   exact-window energy, post-window energy, checksums
              measure_a100_table.py  same protocol as the recorded A100 run, with command-line
                                     arguments and an optional dump of every raw value; the recorded run
                                     used the earlier script without those two additions


7. STILL TO FILL IN (cannot be recovered from the records)
----------------------------------------------------------
[ ] Date of the A100 run (Colab)
[ ] Boot target of the Jetson: the paper's measurement paragraph does not say headless, only the Table 11
    caption did; the word is removed. Restore it only after saving `systemctl get-default`.
