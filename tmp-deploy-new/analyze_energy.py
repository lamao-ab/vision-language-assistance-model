"""
analyze_energy.py — power_log.txt + markers.csv -> paper numbers  (FIXED)
=========================================================================
ROOT-CAUSE FIX:
  tegrastats --logfile stamps every line with WHOLE-SECOND resolution
  (MM-DD-YYYY HH:MM:SS), even though sampling is at 100 ms. The previous version
  parsed at second resolution, so trapezoidal integration over ~1.8 s inference
  windows collapsed: dt=0 for all same-second pairs, dt=1 only at boundaries.
  Energy was under-counted (~half) and quantized run-to-run (huge std), and the
  TOTAL vs NET figures could not reconcile.

  Fix: samples arrive at a known fixed interval, so we RECONSTRUCT sub-second
  timestamps by spreading each second's samples evenly across that second.
  Integration and window assignment then work at true ~100 ms resolution.

Reports:
  P_sys, P_idle, P_active (mean power in each window),
  Energy per query TOTAL and NET-of-idle (trapezoid, Joules),
  Average power at a duty cycle, and battery life.
"""
import re, csv, statistics as st
from collections import OrderedDict
from datetime import datetime

POWER_LOG  = "power_log.txt"
MARKERS    = "markers.csv"
GUARD      = 0.3          # s dropped at each window edge (ramp)
BATTERY_WH = 100.0
QUERIES_HR = 10

# ── parse tegrastats (whole-second epoch, VDD_IN mW) ───────────────────────
raw = []
for line in open(POWER_LOG):
    m = re.search(r"VDD_IN (\d+)mW", line)
    p = line.split()
    if not m or len(p) < 2:
        continue
    try:
        ep = datetime.strptime(p[0] + " " + p[1], "%m-%d-%Y %H:%M:%S").timestamp()
    except ValueError:
        continue
    raw.append((ep, int(m.group(1))))
raw.sort(key=lambda x: x[0])

# ── RECONSTRUCT sub-second timestamps ──────────────────────────────────────
# tegrastats logs whole seconds; spread each second's N samples evenly across it.
by_sec = OrderedDict()
for ep, v in raw:
    by_sec.setdefault(ep, []).append(v)
samples = []
for sec, vals in by_sec.items():
    n = len(vals)
    for j, v in enumerate(vals):
        samples.append((sec + (j + 0.5) / n, v))   # 0.5/n centering -> ~100 ms spacing
samples.sort(key=lambda x: x[0])

def win_samples(t0, t1):
    return [v for t, v in samples if t0 + GUARD <= t <= t1 - GUARD]

def trapz_energy(t0, t1):
    pts = [(t, v) for t, v in samples if t0 <= t <= t1]
    if len(pts) < 2:
        return None, None
    E = 0.0
    for i in range(len(pts) - 1):
        dt = pts[i + 1][0] - pts[i][0]
        E += (pts[i][1] + pts[i + 1][1]) / 2 / 1000.0 * dt   # mW->W * s = J
    return E, pts[-1][0] - pts[0][0]

# ── read markers ───────────────────────────────────────────────────────────
windows, infer = {}, []
for row in csv.DictReader(open(MARKERS)):
    lbl, a, b = row["label"], float(row["t_start"]), float(row["t_end"])
    (infer.append((a, b)) if lbl.startswith("INFER") else windows.__setitem__(lbl, (a, b)))

def mean_power(label):
    if label not in windows:
        return None
    vals = win_samples(*windows[label])
    return st.mean(vals) / 1000 if vals else None

P_sys   = mean_power("IDLE_SYS")
P_idle  = mean_power("IDLE_MODEL")
act_vals = []
for a, b in infer:
    act_vals += win_samples(a, b)
P_active = st.mean(act_vals) / 1000 if act_vals else None

# ── per-query energy ───────────────────────────────────────────────────────
E_tot, E_net = [], []
for a, b in infer:
    E, dur = trapz_energy(a, b)
    if E is None:
        continue
    E_tot.append(E)
    if P_idle is not None:
        E_net.append(E - P_idle * dur)

def pm(x): return f"{st.mean(x):.2f} ± {st.pstdev(x):.2f}" if x else "n/a"

print("=" * 60); print("POWER (W)"); print("=" * 60)
print(f"P_sys   (system idle, no model)   : {P_sys:.2f}"   if P_sys   else "P_sys   : n/a")
print(f"P_idle  (model loaded, idle)      : {P_idle:.2f}"  if P_idle  else "P_idle  : n/a")
print(f"P_active(during inference, mean)  : {P_active:.2f}" if P_active else "P_active: n/a")
if P_sys and P_idle and P_active:
    print("-" * 60)
    print(f"Model static overhead (P_idle-P_sys)   : {P_idle - P_sys:.2f} W")
    print(f"Dynamic inference cost(P_active-P_idle): {P_active - P_idle:.2f} W")

print("\n" + "=" * 60); print("ENERGY PER QUERY (Joules)"); print("=" * 60)
print(f"Total (incl. idle floor) : {pm(E_tot)} J   (n={len(E_tot)})")
print(f"Net of idle (dynamic)    : {pm(E_net)} J")

# ── reconciliation self-check ──────────────────────────────────────────────
if P_active and P_idle and E_tot and infer:
    t_inf = st.mean([b - a for a, b in infer])
    print("-" * 60)
    print("Self-check (should agree to ~5%):")
    print(f"  mean inference duration        : {t_inf:.2f} s")
    print(f"  E_total  vs  P_active*t        : {st.mean(E_tot):.2f}  vs  {P_active*t_inf:.2f} J")
    print(f"  E_net    vs  (P_act-P_idle)*t  : {st.mean(E_net):.2f}  vs  {(P_active-P_idle)*t_inf:.2f} J")

# ── duty cycle & battery ───────────────────────────────────────────────────
if P_idle and P_active and E_tot:
    avg_infer_s = st.mean([b - a for a, b in infer])
    active_s_hr = QUERIES_HR * avg_infer_s
    E_hr = active_s_hr * P_active + (3600 - active_s_hr) * P_idle
    avg_power = E_hr / 3600
    print("\n" + "=" * 60); print(f"DUTY CYCLE & BATTERY ({BATTERY_WH} Wh pack)"); print("=" * 60)
    print(f"Assumption: {QUERIES_HR} queries/hr, {avg_infer_s:.2f}s each")
    print(f"Average power (realistic) : {avg_power:.2f} W")
    print(f"Battery — continuous      : {BATTERY_WH / P_active:.1f} h   (always inferring @P_active)")
    print(f"Battery — realistic       : {BATTERY_WH / avg_power:.1f} h")