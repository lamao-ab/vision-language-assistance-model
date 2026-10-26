#!/usr/bin/env python3
"""
analyze_energy_v3.py  (v3.1)
============================
Regenerates the Jetson power and energy figures from the two raw files written by
energy_run.py:

    power_log.txt   tegrastats log (VDD_IN rail, ~100 ms interval, whole-second
                    wall-clock stamps in LOCAL time)
    markers.csv     window boundaries as epoch seconds: IDLE_SYS, IDLE_MODEL,
                    then one row per query (INFER_0 ... or FULL_0 ...)

What it does differently from analyze_energy.py
-----------------------------------------------
1. Time alignment. The log carries local clock time and the markers carry epoch
   time. The offset is detected automatically by aligning the start of IDLE_SYS
   (energy_run.py starts tegrastats 2 s earlier) with the first log line, rounded
   to a 15-minute time zone step, and printed so it can be checked. Use
   --tz_hours to override.
2. Exact window. Energy is integrated over the full marked window with power
   linearly interpolated at both edges. (analyze_energy.py summed only the samples
   that fall inside each window, which drops up to ~0.1 s per query.)
3. Energy after the window. Power does not return to idle at once, so the energy
   of the rest interval that follows each query (up to the next query) is reported
   separately, together with a fitted decay time constant and an estimate of the
   energy beyond the observed rest. The estimate is labelled as such.

Definitions
-----------
P_sys, P_idle   mean VDD_IN in IDLE_SYS / IDLE_MODEL, trimmed by --guard seconds at
                each window edge
P_active        mean VDD_IN inside the query windows, trimmed the same way
total energy    integral of P dt over the exact window (idle floor included)
net energy      integral of (P - P_idle) dt
cycle           from one query start to the next query start (window + rest)

Usage
-----
    python analyze_energy_v3.py --log power_log.txt --markers markers.csv \
        --out energy_summary_table13_pair.json
"""
import argparse
import csv
import hashlib
import json
import re
import statistics as st
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np

SCRIPT_VERSION = "3.1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_markers(path):
    idle, queries = {}, []
    for r in csv.DictReader(open(path)):
        a, b = float(r["t_start"]), float(r["t_end"])
        if r["label"].startswith("IDLE"):
            idle[r["label"]] = (a, b)
        else:
            queries.append((a, b))
    for needed in ("IDLE_SYS", "IDLE_MODEL"):
        if needed not in idle:
            raise SystemExit(f"{path}: missing marker {needed}")
    if len(queries) < 2:
        raise SystemExit(f"{path}: need at least two query windows")
    return idle, queries


def load_log(path, idle_sys_start, tz_hours=None):
    """Return sample times (epoch s) and power (W). Sub-second times are reconstructed
    by spreading the samples of each wall-clock second evenly across it."""
    stamped = []
    for ln in open(path):
        m, p = re.search(r"VDD_IN (\d+)mW", ln), ln.split()
        if m and len(p) >= 2:
            ep = datetime.strptime(p[0] + " " + p[1], "%m-%d-%Y %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            stamped.append((ep, int(m.group(1))))
    if not stamped:
        raise SystemExit(f"{path}: no VDD_IN samples found")
    off = (idle_sys_start - 2.0) - stamped[0][0]               # marker clock minus log clock
    if tz_hours is None:
        tz = round(off / 900.0) * 900.0
        resid = off - tz
        if abs(resid) > 5:
            raise SystemExit(f"log and markers do not align (residual {resid:.1f} s): different runs?")
    else:
        tz, resid = tz_hours * 3600.0, off - tz_hours * 3600.0
    by = OrderedDict()
    for ep, v in stamped:
        by.setdefault(ep + tz, []).append(v)
    samples = [(sec + (j + 0.5) / len(vs), v) for sec, vs in by.items() for j, v in enumerate(vs)]
    t = np.array([s[0] for s in samples])
    p = np.array([s[1] for s in samples]) / 1000.0
    return t, p, tz / 3600.0, resid


def integrate(t, p, a, b, base=0.0):
    m = (t > a) & (t < b)
    tt = np.concatenate(([a], t[m], [b]))
    pp = np.concatenate(([np.interp(a, t, p)], p[m], [np.interp(b, t, p)])) - base
    return float(np.sum((pp[1:] + pp[:-1]) / 2.0 * np.diff(tt)))


def pm(vals, nd=2):
    return f"{st.mean(vals):.{nd}f} +- {st.pstdev(vals):.{nd}f}"


def analyse(log, markers, tz_hours=None, guard=0.3, battery_wh=100.0, queries_hr=10):
    idle, q = load_markers(markers)
    t, p, tz, resid = load_log(log, idle["IDLE_SYS"][0], tz_hours)

    def window_mean(a, b):
        return float(p[(t >= a + guard) & (t <= b - guard)].mean())

    P_sys, P_idle = window_mean(*idle["IDLE_SYS"]), window_mean(*idle["IDLE_MODEL"])
    P_act = float(np.mean(np.concatenate([p[(t >= a + guard) & (t <= b - guard)] for a, b in q])))
    dur = [b - a for a, b in q]

    e_total = [integrate(t, p, a, b) for a, b in q]
    e_net = [integrate(t, p, a, b, P_idle) for a, b in q]
    n = len(q) - 1                                  # the last query has no following rest interval in the log
    cycle = [integrate(t, p, q[i][0], q[i + 1][0], P_idle) for i in range(n)]
    tail = [cycle[i] - e_net[i] for i in range(n)]
    rest = st.mean([q[i + 1][0] - q[i][1] for i in range(n)])

    centers = np.arange(0.05, min(rest, 1.0), 0.1)  # excess power at fixed offsets after each window
    excess = np.array([np.mean([np.interp(q[i][1] + c, t, p) - P_idle for i in range(n)]) for c in centers])
    ok = excess > 0.05
    tau = float(-1.0 / np.polyfit(centers[ok], np.log(excess[ok]), 1)[0]) if ok.sum() >= 3 else None
    end_excess = float(np.mean([p[(t >= q[i + 1][0] - 0.2) & (t < q[i + 1][0])].mean() - P_idle for i in range(n)]))
    beyond = end_excess * tau if tau else None
    complete = st.mean(cycle) + (beyond or 0.0)
    avg_w = P_idle + queries_hr * complete / 3600.0

    return {
        "script_version": SCRIPT_VERSION,
        "inputs": {"power_log": {"file": log, "sha256": sha256(log), "samples": int(len(t))},
                   "markers": {"file": markers, "sha256": sha256(markers), "queries": len(q)}},
        "parameters": {"guard_s": guard, "battery_wh": battery_wh, "queries_per_hour": queries_hr},
        "alignment": {"timezone_offset_h": tz, "residual_s": round(resid, 2)},
        "windows": {"n_queries": len(q), "window_s": f"{st.mean(dur):.3f} +- {st.pstdev(dur):.4f}",
                    "rest_s": round(rest, 2)},
        "power_W": {"P_sys": round(P_sys, 2), "P_idle": round(P_idle, 2), "P_active": round(P_act, 2)},
        "energy_per_query_J": {
            "window_total_incl_idle_floor": pm(e_total),
            "window_net": pm(e_net),
            "tail_net_observed_rest": pm(tail),
            "cycle_net_observed": pm(cycle),
            "beyond_rest_estimate": None if beyond is None else round(beyond, 2),
            "complete_net_estimate": round(complete, 2)},
        "post_window_decay": {
            "time_constant_s": None if tau is None else round(tau, 2),
            "excess_W_at_end_of_rest": round(end_excess, 2),
            "excess_W_at_offsets_0.05_0.15_s": [round(float(x), 2) for x in excess]},
        "illustrative_battery_100Wh": {
            "average_power_W_at_10qph": round(avg_w, 2),
            "hours_at_10qph": round(battery_wh / avg_w, 1),
            "hours_continuous_at_P_active": round(battery_wh / P_act, 1)},
    }


def report(r):
    w, p, e, d, b = r["windows"], r["power_W"], r["energy_per_query_J"], r["post_window_decay"], r["illustrative_battery_100Wh"]
    print(f"analyze_energy_v3.py v{r['script_version']}")
    print(f"log     : {r['inputs']['power_log']['file']}  ({r['inputs']['power_log']['samples']} samples)")
    print(f"markers : {r['inputs']['markers']['file']}  ({w['n_queries']} query windows)")
    print(f"clock alignment: log is {r['alignment']['timezone_offset_h']:+.2f} h from epoch time "
          f"(residual {r['alignment']['residual_s']} s)\n")
    print(f"POWER (W)       P_sys {p['P_sys']:.2f} | P_idle {p['P_idle']:.2f} | P_active {p['P_active']:.2f}")
    print(f"WINDOW          {w['window_s']} s, followed by {w['rest_s']} s rest\n")
    print("ENERGY PER QUERY (J)")
    print(f"  total, exact window (idle floor included) : {e['window_total_incl_idle_floor']}")
    print(f"  net above idle, exact window              : {e['window_net']}")
    print(f"  net above idle, following rest interval   : {e['tail_net_observed_rest']}")
    print(f"  net above idle, window + rest             : {e['cycle_net_observed']}")
    print(f"  net above idle, complete (extrapolated)   : {e['complete_net_estimate']}  <- estimate\n")
    print(f"POST-WINDOW DECAY  time constant {d['time_constant_s']} s | excess at end of rest {d['excess_W_at_end_of_rest']} W")
    print(f"ILLUSTRATIVE BATTERY (100 Wh, 10 queries/h): {b['hours_at_10qph']} h | continuous at P_active: {b['hours_continuous_at_P_active']} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="power_log.txt")
    ap.add_argument("--markers", default="markers.csv")
    ap.add_argument("--out", default="energy_summary.json")
    ap.add_argument("--tz_hours", type=float, default=None, help="override the detected clock offset")
    ap.add_argument("--guard", type=float, default=0.3, help="seconds trimmed at window edges for power means")
    ap.add_argument("--battery_wh", type=float, default=100.0)
    ap.add_argument("--queries_hr", type=float, default=10)
    a = ap.parse_args()
    r = analyse(a.log, a.markers, a.tz_hours, a.guard, a.battery_wh, a.queries_hr)
    json.dump(r, open(a.out, "w"), indent=2)
    report(r)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
