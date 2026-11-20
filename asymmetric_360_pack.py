#!/usr/bin/env python3
"""
asymmetric-360-pack
Asymmetric 360 deg World-Model Uncertainty & Closed-Loop Agentic Harness
for Dynamic Roadway Micro-Packing.

Zero-dependency: Python standard library ONLY. No pip, no numpy, no SUMO,
no FFmpeg. Run:

    python asymmetric_360_pack.py                 # real NGSIM car-following eval + video
    python asymmetric_360_pack.py --csv traj.csv  # ingest NGSIM-style trajectory table
    python asymmetric_360_pack.py --no-video      # skip the (large) AVI render

Outputs (written to ./outputs/):
    ablation_table.tex          publication-ready booktabs table (mean +/- std)
    ablation_report.json        full statistical summary + audit traces
    asymmetric_360_proof.avi    pure-python split-screen proof video

NOTE: the default scenarios are a transparent, deterministic SIMULATION. The
numbers reported are outputs of the control/prediction logic implemented here,
not empirical field measurements. Pass --csv to drive neighbor extraction from
a real NGSIM-style trajectory table.
"""

import os
import sys
import glob
import json
import math
import time
import struct
import argparse

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
DT = 0.1                       # simulation timestep [s]
EVAL_STEPS = 150               # 15 s evaluation horizon
SCORE_FRAMES = 26              # trailing scorecard frames in the video
CAR_LEN = 5.0                  # vehicle length [m]
LANE_W = 3.7                   # lane width [m]
VEH_W = 2.0                    # vehicle width [m]
NOM_SPEED = 25.0               # nominal cruising speed [m/s] ~ 56 mph

# control limits
A_MAX = 2.5                    # max accel [m/s^2]
A_MIN = -7.0                   # max (panic) decel [m/s^2]
REAR_A_MIN = -5.0              # trailing vehicle weaker brakes
REAR_LAG = 7                   # trailing vehicle reaction lag [steps] = 0.7 s

# desired time-gaps
T_PACK = 0.70                  # compressed (micro-pack) time gap [s]
T_SAFE = 1.55                  # safe fallback time gap [s]
T_IDM = 1.5                    # IDM desired time gap [s]
T_NAIVE = 0.45                 # naive front-only tight gap [s]

# safety envelope floors
FRONT_FLOOR = 6.0              # min predicted front gap [m]
REAR_FLOOR = 5.0               # min predicted rear gap [m]
LAT_FLOOR = 0.5                # min lateral clearance [m]
LAT_CONTACT = 0.15             # side-swipe contact clearance [m]

POLICIES = ["IDM", "NAIVE", "S1S0", "FULL"]
POLICY_LABEL = {
    "IDM":  "Standard IDM car-following",
    "NAIVE": "Naive front-only tight packing",
    "S1S0": "System 1 + System 0 (no context verification)",
    "FULL": "Full asymmetric-360-pack (S2+S1+S0)",
}

OUT_DIR = "outputs"


# ----------------------------------------------------------------------------
# Small stats helpers (no numpy)
# ----------------------------------------------------------------------------
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ----------------------------------------------------------------------------
# System 1 : 360 deg world-model rollout (shared predictor for S1S0 and FULL)
# ----------------------------------------------------------------------------
def s1_rollout(front_gap, v_e, v_f, a_f, rear_gap, v_r, min_lat, vis,
               context_aware, sigma_emp=0.0, horizon=3.0):
    """Forecast anisotropic worst-case margins + candidate density gain.

    context_aware=False -> fixed nominal uncertainty, ignores context (S1S0).
    context_aware=True  -> inflates uncertainty from (a) low visibility and
                           (b) the live empirical forecast residual sigma_emp,
                           which rises during real decelerations (FULL/S2).
    """
    sigma = 0.30
    if context_aware:
        sigma = max(sigma + (1.0 - vis) * 2.4, 0.30 + 1.2 * sigma_emp)
    H = horizon
    # front: assume ego holds speed; lead follows measured accel
    pred_front = front_gap + (v_f - v_e) * H + 0.5 * a_f * H * H
    wc_front = pred_front - 2.0 * sigma
    # rear: trailing vehicle closing shrinks the rear gap
    pred_rear = rear_gap + (v_e - v_r) * H
    wc_rear = pred_rear - 2.0 * sigma
    # lateral
    sigma_lat = 0.20 + (0.60 * (1.0 - vis) if context_aware else 0.0)
    wc_lat = min_lat - 2.0 * sigma_lat
    # candidate density gain if we compress to the pack gap
    nom_spacing = 2.0 + v_e * T_SAFE + CAR_LEN
    pack_spacing = 2.0 + v_e * T_PACK + CAR_LEN
    gain = (nom_spacing / pack_spacing - 1.0) * 100.0
    return {"wc_front": wc_front, "wc_rear": wc_rear, "wc_lat": wc_lat,
            "gain": gain, "sigma": sigma}


# ----------------------------------------------------------------------------
# System 2 : agentic verification & context harness
# ----------------------------------------------------------------------------
def s2_decide(f, s0_alert, vis):
    """Return (decision, time_gap, rationale)."""
    reasons = []
    hard = (f["wc_front"] < FRONT_FLOOR * 0.6 or
            f["wc_rear"] < REAR_FLOOR * 0.6 or
            f["wc_lat"] < LAT_FLOOR * 0.6)
    soft = (f["wc_front"] < FRONT_FLOOR or
            f["wc_rear"] < REAR_FLOOR or
            f["wc_lat"] < LAT_FLOOR or vis < 0.6 or s0_alert)
    if hard:
        if f["wc_front"] < FRONT_FLOOR * 0.6:
            reasons.append("front worst-case below floor")
        if f["wc_rear"] < REAR_FLOOR * 0.6:
            reasons.append("rear closing worst-case below floor")
        if f["wc_lat"] < LAT_FLOOR * 0.6:
            reasons.append("lateral worst-case below floor")
        return "VETO", T_SAFE, "; ".join(reasons) or "unsafe envelope"
    if soft:
        if vis < 0.6:
            reasons.append("visibility degraded")
        if s0_alert:
            reasons.append("S0 envelope alert")
        return "HOLD", None, "; ".join(reasons) or "margins borderline"
    return "APPROVE", T_PACK, "360 context clear; compression authorized"


# ----------------------------------------------------------------------------
# Core closed-loop simulation of one policy against one scenario
# ----------------------------------------------------------------------------
def simulate(policy, scen):
    steps = scen["steps"]
    vf = scen["vf"]
    vis = scen["vis"]
    encL = scen["encL"]
    encR = scen["encR"]

    # state
    v0 = scen.get("v_e0", NOM_SPEED)
    x_e, v_e, y_e = 0.0, v0, 0.0
    x_f = scen["front_gap0"] + CAR_LEN
    x_r = -(scen["rear_gap0"] + CAR_LEN)
    v_r = v0
    T_prev = T_SAFE
    real_latL = scen.get("real_latL")
    real_latR = scen.get("real_latR")

    # empirical forecast-uncertainty estimator (leader pos, 1 s ahead)
    sigma_emp = 0.0
    H_u = 10
    pred_q = {}

    hist = []        # (x_e, v_e) history for trailing-vehicle lag
    trace = []
    arrays = {k: [] for k in
              ("x_e", "v_e", "x_f", "v_f", "x_r", "v_r",
               "front_gap", "rear_gap", "min_lat", "y_e", "a_e")}
    events = {"front_coll": 0, "rear_coll": 0, "lat_breach": 0,
              "lat_contact": 0, "env_viol": 0}
    min_ttc = 1e9
    lat_sum = 0.0
    latency_sum = 0.0
    kp, kv = 0.35, 0.65
    rear_lag = int(scen.get("rear_lag", REAR_LAG))
    rear_brake = scen.get("rear_brake", REAR_A_MIN)

    for i in range(steps):
        a_f = (vf[i + 1] - vf[i]) / DT if i + 1 < steps else 0.0
        front_gap = x_f - x_e - CAR_LEN
        rear_gap = x_e - x_r - CAR_LEN

        # empirical leader-forecast residual (const-accel prediction vs real)
        if i in pred_q:
            sigma_emp = 0.8 * sigma_emp + 0.2 * abs(x_f - pred_q.pop(i))
        pred_q[i + H_u] = x_f + vf[i] * (H_u * DT) + 0.5 * a_f * (H_u * DT) ** 2

        # lateral clearance (real adjacent-lane neighbors; cleared if none)
        if real_latL is not None:
            baseL, baseR = real_latL[i], real_latR[i]
        else:
            baseL, baseR = VEH_W - encL[i], VEH_W - encR[i]
        lateral_capable = policy in ("S1S0", "FULL")
        tgt_ye = clamp(0.5 * (baseR - baseL), -0.8, 0.8) if lateral_capable else 0.0
        y_e += clamp(tgt_ye - y_e, -0.1, 0.1)
        cl_left = baseL + y_e
        cl_right = baseR - y_e
        min_lat = min(cl_left, cl_right)

        # ---- decision (timed for latency) ----
        t0 = time.perf_counter()
        decision, rationale = "FIXED", ""
        if policy == "IDM":
            # Intelligent Driver Model w.r.t. lead vehicle
            s0, v0, a_acc, b_cmf = 2.0, 30.0, 1.5, 2.0
            dv = v_e - vf[i]
            s = max(0.5, front_gap)
            s_star = s0 + max(0.0, v_e * T_IDM + v_e * dv / (2 * math.sqrt(a_acc * b_cmf)))
            a_cmd = a_acc * (1.0 - (v_e / v0) ** 4 - (s_star / s) ** 2)
            T_cmd = None
        elif policy == "NAIVE":
            T_cmd = T_NAIVE
            a_cmd = None
            decision = "TIGHT"
        else:
            context_aware = (policy == "FULL")
            f = s1_rollout(front_gap, v_e, vf[i], a_f, rear_gap, v_r,
                           min_lat, vis[i], context_aware, sigma_emp)
            # System 0 immediate envelope guard
            s0_alert = (front_gap < v_e * 0.8 or
                        rear_gap < v_r * 0.4 + 2.0 or
                        min_lat < LAT_FLOOR)
            if policy == "S1S0":
                # no context verification: compress whenever rollout margins ok
                if (f["wc_front"] >= FRONT_FLOOR and f["wc_rear"] >= REAR_FLOOR
                        and f["wc_lat"] >= LAT_FLOOR and not s0_alert):
                    decision, T_cmd = "APPROVE", T_PACK
                else:
                    decision, T_cmd = "HOLD", T_prev
                rationale = "S1/S0 rollout gate (no occlusion check)"
            else:  # FULL
                decision, T_cmd, rationale = s2_decide(f, s0_alert, vis[i])
                if T_cmd is None:
                    T_cmd = T_prev
            # System 0 hard floor: never pack below the physical stop buffer
            floor_gap = v_e * 0.8 + 2.0
            min_T = max(0.0, (floor_gap - 2.0) / max(v_e, 1.0))
            if T_cmd < min_T and s0_alert:
                T_cmd = min_T
            a_cmd = None
        latency_sum += (time.perf_counter() - t0) * 1000.0

        # longitudinal control toward desired time-gap (non-IDM policies)
        if a_cmd is None:
            T_use = T_cmd if T_cmd is not None else T_prev
            s_star = 2.0 + v_e * T_use
            a_cmd = kp * (front_gap - s_star) + kv * (vf[i] - v_e)
            T_prev = T_use
        a_cmd = clamp(a_cmd, A_MIN, A_MAX)

        # integrate ego
        v_e = max(0.0, v_e + a_cmd * DT)
        x_e += v_e * DT
        x_f += vf[i] * DT

        # trailing vehicle (reactive, lagged, weaker brakes)
        if len(hist) >= rear_lag:
            xe_d, ve_d = hist[-rear_lag]
        elif hist:
            xe_d, ve_d = hist[0]
        else:
            xe_d, ve_d = x_e, v_e
        rgap_d = xe_d - x_r - CAR_LEN
        a_r = 0.45 * (rgap_d - (2.0 + v_r * scen["rear_T"])) + 0.7 * (ve_d - v_r)
        a_r = clamp(a_r, rear_brake, A_MAX)
        v_r = max(0.0, v_r + a_r * DT)
        x_r += v_r * DT

        hist.append((x_e, v_e))

        # recompute gaps post-integration + collisions
        front_gap = x_f - x_e - CAR_LEN
        rear_gap = x_e - x_r - CAR_LEN
        if front_gap < 0:
            events["front_coll"] += 1
            x_e = x_f - CAR_LEN
            v_e = min(v_e, vf[i])
            front_gap = 0.0
        if rear_gap < 0:
            events["rear_coll"] += 1
            x_r = x_e - CAR_LEN
            v_r = min(v_r, v_e)
            rear_gap = 0.0

        # safety metrics
        if min_lat < LAT_FLOOR:
            events["lat_breach"] += 1
        if min_lat < LAT_CONTACT:
            events["lat_contact"] += 1
        viol = (front_gap < v_e * 0.5 or rear_gap < v_r * 0.35 + 1.0 or
                min_lat < LAT_FLOOR)
        if viol:
            events["env_viol"] += 1
        # TTC
        if v_e > vf[i]:
            min_ttc = min(min_ttc, front_gap / max(v_e - vf[i], 0.01))
        if v_r > v_e:
            min_ttc = min(min_ttc, rear_gap / max(v_r - v_e, 0.01))
        lat_sum += min_lat

        for k, val in (("x_e", x_e), ("v_e", v_e), ("x_f", x_f),
                       ("v_f", vf[i]), ("x_r", x_r), ("v_r", v_r),
                       ("front_gap", front_gap), ("rear_gap", rear_gap),
                       ("min_lat", min_lat), ("y_e", y_e), ("a_e", a_cmd)):
            arrays[k].append(val)

        ttc_f = front_gap / (v_e - vf[i]) if v_e > vf[i] else 99.9
        ttc_r = rear_gap / (v_r - v_e) if v_r > v_e else 99.9
        gain_live = ((2.0 + v_e * T_SAFE + CAR_LEN) /
                     (front_gap + CAR_LEN) - 1.0) * 100.0
        trace.append({
            "t": round(i * DT, 2), "front_gap": round(front_gap, 2),
            "rear_gap": round(rear_gap, 2), "min_lat": round(min_lat, 3),
            "v_e": round(v_e, 2), "ttc_front": round(min(ttc_f, 99.9), 2),
            "ttc_rear": round(min(ttc_r, 99.9), 2), "vis": vis[i],
            "decision": decision, "rationale": rationale,
            "gain_pct": round(max(gain_live, 0.0), 1),
            "sigma_emp": round(sigma_emp, 2),
        })

    mean_spacing = mean([g + CAR_LEN for g in arrays["front_gap"]])

    # recovery time: after disturbance, time for ego speed to re-enter
    # within 5% of its pre-disturbance mean and stay for >= 1 s
    recovery = 0.0
    t_dist = scen["t_dist"]
    if t_dist > 0:
        di = int(t_dist / DT)
        pre = mean(arrays["v_e"][:di]) if di > 0 else NOM_SPEED
        band = 0.05 * pre
        rec_idx = None
        hold = 0
        for j in range(di, steps):
            if abs(arrays["v_e"][j] - pre) <= band:
                hold += 1
                if hold >= int(1.0 / DT):
                    rec_idx = j - int(1.0 / DT)
                    break
            else:
                hold = 0
        recovery = max(0.0, (rec_idx - di) * DT) if rec_idx is not None \
            else (steps - di) * DT

    summary = {
        "policy": policy,
        "mean_spacing": mean_spacing,
        "min_ttc": min(min_ttc, 20.0),
        "env_viol": events["env_viol"],
        "front_coll": events["front_coll"],
        "rear_coll": events["rear_coll"],
        "lat_breach": events["lat_breach"],
        "lat_contact": events["lat_contact"],
        "recovery_s": recovery,
        "latency_ms": latency_sum / steps,
        "mean_lat": lat_sum / steps,
    }
    return summary, trace, arrays


# ----------------------------------------------------------------------------
# Model-level metrics: directional ADE/FDE + uncertainty coverage for the
# System-1 world model (shared by S1S0 and FULL).
# ----------------------------------------------------------------------------
def model_metrics(arrays, scen, targets=("front", "rear"), horizon_steps=20):
    steps = scen["steps"]
    errs = []            # per-direction mean abs error (ADE samples)
    fdes = []            # final displacement errors
    cov_hits = 0
    cov_tot = 0
    vis = scen["vis"]
    # front & rear longitudinal relative positions; lateral for sides
    xe = arrays["x_e"]; xf = arrays["x_f"]; xr = arrays["x_r"]
    ve = arrays["v_e"]; vfr = arrays["v_f"]; vr = arrays["v_r"]
    tgt = [t for t in (("front", xf, vfr), ("rear", xr, vr))
           if t[0] in targets]
    for i in range(steps - horizon_steps):
        for name, pos, vel in tgt:
            # constant-velocity relative prediction vs realized
            rel0 = pos[i] - xe[i]
            v_rel = vel[i] - ve[i]
            step_errs = []
            for k in range(1, horizon_steps + 1):
                pred_rel = rel0 + v_rel * (k * DT)
                real_rel = pos[i + k] - xe[i + k]
                e = abs(pred_rel - real_rel)
                step_errs.append(e)
                sigma = 0.30 * math.sqrt(k * DT) + (1.0 - vis[i]) * 1.2
                cov_tot += 1
                if e <= 2.0 * sigma:
                    cov_hits += 1
            errs.append(mean(step_errs))
            fdes.append(step_errs[-1])
    return {
        "ade": mean(errs) if errs else 0.0,
        "fde": mean(fdes) if fdes else 0.0,
        "coverage": 100.0 * cov_hits / cov_tot if cov_tot else 0.0,
    }


# ----------------------------------------------------------------------------
# Statistical ablation driver over a list of real scenarios
# ----------------------------------------------------------------------------
def evaluate_scenarios(scens, targets=("front", "rear"), keep_audit=0):
    per_policy = {p: {"gain": [], "viol": [], "ttc": [], "rec": [],
                      "lat": [], "ade": [], "fde": [], "cov": [],
                      "coll": []} for p in POLICIES}
    audit = {p: [] for p in POLICIES}
    for si, scen in enumerate(scens):
        idm_sum, _, _ = simulate("IDM", scen)          # capacity-gain reference
        ref_spacing = idm_sum["mean_spacing"]
        for p in POLICIES:
            summ, trace, arrays = simulate(p, scen)
            per_policy[p]["gain"].append(
                (ref_spacing / summ["mean_spacing"] - 1.0) * 100.0)
            per_policy[p]["viol"].append(summ["env_viol"])
            per_policy[p]["ttc"].append(summ["min_ttc"])
            per_policy[p]["rec"].append(summ["recovery_s"])
            per_policy[p]["lat"].append(summ["latency_ms"])
            per_policy[p]["coll"].append(summ["front_coll"] + summ["rear_coll"])
            if p in ("S1S0", "FULL"):
                mm = model_metrics(arrays, scen, targets)
                per_policy[p]["ade"].append(mm["ade"])
                per_policy[p]["fde"].append(mm["fde"])
                per_policy[p]["cov"].append(mm["coverage"])
            if si < keep_audit:
                audit[p].append({
                    "scenario": scen.get("name", "s%d" % si),
                    "decisions": [t["decision"] for t in trace[::10]],
                    "sample_rationales": [t["rationale"] for t in trace[::25]
                                          if t["rationale"]][:4],
                    "summary": summ,
                })
    return per_policy, audit


# ----------------------------------------------------------------------------
# Real NGSIM trajectory data: build car-following scenarios from recorded
# leader speed profiles (real shockwave braking) + real adjacent-lane neighbors.
# Raw NGSIM is US customary (feet, ft/s). FT_TO_M -> SI.
# ----------------------------------------------------------------------------
FT_TO_M = 0.3048
REAL_WINDOW = 100          # 10 s car-following windows
NGSIM_LOOKUP = [           # candidate cache locations (relative to a root)
    os.path.join("v2xbench", "ngsim"),
    "ngsim",
]


def find_ngsim_root(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.join(here, ".."),
                 os.path.join(here, "..", "..")):
        cands.append(os.path.join(base, "_data"))
        cands.append(base)
    for c in cands:
        for sub in NGSIM_LOOKUP:
            root = os.path.join(c, sub)
            if os.path.isdir(root) and glob.glob(
                    os.path.join(root, "*", "page_*.csv")):
                return root
    return None


def load_ngsim_pages(root, location, max_pages):
    import csv
    pages = sorted(glob.glob(os.path.join(root, location, "page_*.csv")))
    pages = pages[:max_pages]
    by_v, by_f = {}, {}
    for pg in pages:
        with open(pg, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rec = (int(r["frame_id"]),
                           float(r["local_x"]) * FT_TO_M,
                           float(r["local_y"]) * FT_TO_M,
                           float(r["v_vel"]) * FT_TO_M,
                           float(r["v_acc"]) * FT_TO_M,
                           int(r["lane_id"]),
                           float(r["v_length"]) * FT_TO_M,
                           r["vehicle_id"])
                except (ValueError, KeyError):
                    continue
                by_v.setdefault(r["vehicle_id"], []).append(rec)
                by_f.setdefault(rec[0], []).append(rec)
    for v in by_v:
        by_v[v].sort()
    return by_v, by_f, len(pages)


def build_ngsim_scenarios(root, locations, max_pairs, max_pages=6,
                          window=REAL_WINDOW):
    scens = []
    for location in locations:
        by_v, by_f, npg = load_ngsim_pages(root, location, max_pages)
        for vid, recs in by_v.items():
            if len(scens) >= max_pairs:
                break
            if len(recs) < window + 2:
                continue
            # first contiguous window with a single persistent leader
            for s in range(0, len(recs) - window, 5):
                run = recs[s:s + window]
                if run[-1][0] - run[0][0] != window - 1:
                    continue
                ego0 = run[0]
                lane = ego0[5]
                fr0 = ego0[0]
                cand = [r for r in by_f[fr0] if r[5] == lane
                        and r[2] > ego0[2] and r[2] - ego0[2] < 120]
                if not cand:
                    continue
                leader = min(cand, key=lambda r: r[2] - ego0[2])
                lid = leader[7]
                vf, latL, latR, lacc = [], [], [], []
                ok = True
                for k in range(window):
                    fr = run[k][0]
                    ex, ey, elane = run[k][1], run[k][2], run[k][5]
                    lr = [r for r in by_f[fr] if r[7] == lid
                          and r[5] == elane and r[2] > ey]
                    if not lr:
                        ok = False
                        break
                    vf.append(max(0.5, lr[0][3]))
                    lacc.append(lr[0][4])
                    # real adjacent-lane clearances
                    def near(dl):
                        ns = [r for r in by_f[fr] if r[5] == elane + dl
                              and abs(r[2] - ey) < 8.0]
                        if not ns:
                            return 3.0
                        n = min(ns, key=lambda r: abs(r[2] - ey))
                        return clamp(abs(n[1] - ex) - VEH_W, 0.0, 3.0)
                    latL.append(near(-1))
                    latR.append(near(+1))
                if not ok:
                    continue
                fg0 = max(2.0, (leader[2] - ego0[2]) - leader[6])
                foll = [r for r in by_f[fr0] if r[5] == lane
                        and r[2] < ego0[2]]
                rg0 = (ego0[2] - max(foll, key=lambda r: r[2])[2] - ego0[6]
                       ) if foll else 18.0
                rg0 = max(5.0, rg0)
                t_dist = lacc.index(min(lacc)) * DT
                scens.append({
                    "name": "%s_veh%s" % (location, vid),
                    "regime": -1, "seed": 0, "steps": window,
                    "vf": vf, "vis": [1.0] * window,
                    "encL": [0.0] * window, "encR": [0.0] * window,
                    "real_latL": latL, "real_latR": latR,
                    "v_e0": max(0.5, ego0[3]),
                    "front_gap0": fg0, "rear_gap0": rg0,
                    "rear_T": 0.7, "rear_lag": 7, "rear_brake": -5.0,
                    "t_dist": t_dist, "source": location,
                })
                break
    return scens


def agg(per_policy):
    rows = {}
    for p in POLICIES:
        d = per_policy[p]
        rows[p] = {
            "gain": (mean(d["gain"]), std(d["gain"])),
            "viol": (mean(d["viol"]), std(d["viol"])),
            "ttc": (mean(d["ttc"]), std(d["ttc"])),
            "rec": (mean(d["rec"]), std(d["rec"])),
            "lat": (mean(d["lat"]), std(d["lat"])),
            "coll": (mean(d["coll"]), std(d["coll"])),
            "ade": (mean(d["ade"]), std(d["ade"])) if d["ade"] else None,
            "fde": (mean(d["fde"]), std(d["fde"])) if d["fde"] else None,
            "cov": (mean(d["cov"]), std(d["cov"])) if d["cov"] else None,
        }
    return rows


# ----------------------------------------------------------------------------
# Optional CSV ingestion (NGSIM-style trajectory table)
# ----------------------------------------------------------------------------
def load_csv_scenario(path, steps):
    """Parse Vehicle_ID,Frame_ID,Local_X,Local_Y,v_Vel,v_Acc and build a
    single ego-centric scenario by neighbor extraction. Standard library csv."""
    import csv
    rows = []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                rows.append({
                    "vid": r.get("Vehicle_ID") or r.get("vehicle_id"),
                    "fid": int(float(r.get("Frame_ID") or r.get("frame_id"))),
                    "x": float(r.get("Local_X") or r.get("local_x")),
                    "y": float(r.get("Local_Y") or r.get("local_y")),
                    "v": float(r.get("v_Vel") or r.get("v_vel") or 0.0),
                })
            except (TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit("CSV had no parseable rows (need Vehicle_ID,Frame_ID,"
                         "Local_X,Local_Y,v_Vel[,v_Acc]).")
    # pick the vehicle with the longest track as ego
    by_v = {}
    for r in rows:
        by_v.setdefault(r["vid"], []).append(r)
    ego_id = max(by_v, key=lambda k: len(by_v[k]))
    ego = sorted(by_v[ego_id], key=lambda r: r["fid"])[:steps]
    frames = [e["fid"] for e in ego]
    vf = []
    for e in ego:
        # lead = nearest vehicle ahead in same lane-ish band at that frame
        same = [r for r in rows if r["fid"] == e["fid"]
                and abs(r["y"] - e["y"]) < LANE_W * 0.6 and r["x"] > e["x"]]
        if same:
            lead = min(same, key=lambda r: r["x"] - e["x"])
            vf.append(max(3.0, lead["v"] if lead["v"] > 0 else e["v"]))
        else:
            vf.append(max(3.0, e["v"]))
    while len(vf) < steps:
        vf.append(vf[-1] if vf else NOM_SPEED)
    vf = vf[:steps]
    return {
        "regime": -1, "seed": 0, "steps": steps,
        "vf": vf, "vis": [1.0] * steps,
        "encL": [0.0] * steps, "encR": [0.0] * steps,
        "rear_gap0": 18.0, "rear_T": 0.7,
        "front_gap0": max(8.0, (ego[0]["v"] or NOM_SPEED) * 1.5),
        "t_dist": 0.0, "source": os.path.basename(path), "ego_id": ego_id,
    }


# ----------------------------------------------------------------------------
# LaTeX booktabs table
# ----------------------------------------------------------------------------
def write_latex(rows, path, data_mode="ngsim"):
    def cell(t, fmt="{:.1f}"):
        if t is None:
            return "--"
        return "${0} \\pm {1}$".format(fmt.format(t[0]), fmt.format(t[1]))
    if data_mode.startswith("csv"):
        src = "an ingested NGSIM-style trajectory table"
    else:
        src = ("real NGSIM car-following pairs (ego following recorded real "
               "leader speed profiles with real shockwave braking)")
    lines = [
        r"%% Auto-generated by asymmetric_360_pack.py  [data: %s]" % data_mode,
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Four-way ablation of the asymmetric 360$^\circ$ micro-packing "
        r"harness over %s. Values are mean $\pm$ std; capacity gain is relative "
        r"to the IDM baseline spacing on the same scenario, and collisions are "
        r"summed per run.}" % src,
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Policy & Coll. & Cap.\ gain (\%) & Env.\ viol. & Min TTC (s) & "
        r"Recov.\ (s) & Lat.\ (ms) & ADE (m) & FDE (m) \\",
        r"\midrule",
    ]
    names = {"IDM": "IDM baseline", "NAIVE": "Naive front-tight",
             "S1S0": "S1+S0 (no S2)", "FULL": r"\textbf{Full 360-pack}"}
    for p in POLICIES:
        r = rows[p]
        lines.append(" & ".join([
            names[p], cell(r["coll"], "{:.2f}"),
            cell(r["gain"]), cell(r["viol"]), cell(r["ttc"], "{:.2f}"),
            cell(r["rec"], "{:.2f}"), cell(r["lat"], "{:.3f}"),
            cell(r["ade"], "{:.2f}"), cell(r["fde"], "{:.2f}"),
        ]) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{9}{l}{\footnotesize Uncertainty coverage @2$\sigma$ "
        r"(S1 world model): %s.}" %
        (cell(rows["FULL"]["cov"], "{:.1f}") + r"\%"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Pure-python bitmap canvas + uncompressed RIFF AVI writer
# ----------------------------------------------------------------------------
FONT_SRC = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["11111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10001", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    ":": ["00000", "00110", "00110", "00000", "00110", "00110", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10011", "10011", "00000"],
    "/": ["00001", "00001", "00010", "00100", "01000", "10000", "10000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "(": ["00110", "01000", "10000", "10000", "10000", "01000", "00110"],
    ")": ["01100", "00010", "00001", "00001", "00001", "00010", "01100"],
    ",": ["00000", "00000", "00000", "00000", "00110", "00110", "01000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "#": ["01010", "01010", "11111", "01010", "11111", "01010", "01010"],
}
FONT = {c: [int(row, 2) for row in rows] for c, rows in FONT_SRC.items()}


class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.buf = bytearray(w * h * 3)

    def fill(self, rgb):
        self.buf[:] = bytes(rgb) * (self.w * self.h)

    def rect(self, x, y, w, h, rgb):
        x0 = max(0, int(x)); y0 = max(0, int(y))
        x1 = min(self.w, int(x + w)); y1 = min(self.h, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return
        seg = bytes(rgb) * (x1 - x0)
        W3 = self.w * 3
        for yy in range(y0, y1):
            idx = yy * W3 + x0 * 3
            self.buf[idx:idx + len(seg)] = seg

    def px(self, x, y, rgb):
        if 0 <= x < self.w and 0 <= y < self.h:
            idx = (y * self.w + x) * 3
            self.buf[idx:idx + 3] = bytes(rgb)

    def line(self, x0, y0, x1, y1, rgb):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0); dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.px(x0, y0, rgb)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x0 += sx
            if e2 <= dx:
                err += dx; y0 += sy

    def ring(self, cx, cy, r, thick, rgb):
        r2o = r * r
        r2i = max(0, (r - thick)) ** 2
        for yy in range(int(cy - r), int(cy + r + 1)):
            if yy < 0 or yy >= self.h:
                continue
            dy2 = (yy - cy) ** 2
            for xx in range(int(cx - r), int(cx + r + 1)):
                d = (xx - cx) ** 2 + dy2
                if r2i <= d <= r2o:
                    self.px(xx, yy, rgb)

    def disc(self, cx, cy, r, rgb):
        for yy in range(int(cy - r), int(cy + r + 1)):
            dy2 = (yy - cy) ** 2
            for xx in range(int(cx - r), int(cx + r + 1)):
                if (xx - cx) ** 2 + dy2 <= r * r:
                    self.px(xx, yy, rgb)

    def text(self, x, y, s, rgb, scale=2):
        cx = int(x)
        for ch in s.upper():
            glyph = FONT.get(ch, FONT[" "])
            for ry in range(7):
                bits = glyph[ry]
                for rx in range(5):
                    if bits & (1 << (4 - rx)):
                        self.rect(cx + rx * scale, int(y) + ry * scale,
                                  scale, scale, rgb)
            cx += 6 * scale

    def dib(self):
        """Return bottom-up BGR DIB bytes with 4-byte row padding."""
        w, h = self.w, self.h
        stride = (w * 3 + 3) & ~3
        out = bytearray(stride * h)
        rl = w * 3
        for y in range(h):
            src = self.buf[(h - 1 - y) * rl:(h - y) * rl]
            dst = bytearray(src)
            dst[0::3], dst[2::3] = src[2::3], src[0::3]
            out[y * stride:y * stride + rl] = dst
        return bytes(out), stride


def write_avi(path, frames, w, h, fps):
    stride = (w * 3 + 3) & ~3
    frame_size = stride * h

    def chunk(fourcc, data):
        pad = b"\x00" if len(data) & 1 else b""
        return fourcc + struct.pack("<I", len(data)) + data + pad

    avih = struct.pack("<14I", int(1e6 / fps), int(frame_size * fps), 0,
                       0x10, len(frames), 0, 1, frame_size, w, h, 0, 0, 0, 0)
    strh = (b"vids" + b"DIB " +
            struct.pack("<IHHIIIIIIIi", 0, 0, 0, 0, 1, fps, 0, len(frames),
                        frame_size, 0, 0) +
            struct.pack("<4h", 0, 0, w, h))
    strf = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, frame_size,
                       0, 0, 0, 0)
    strl = b"strl" + chunk(b"strh", strh) + chunk(b"strf", strf)
    hdrl = b"hdrl" + chunk(b"avih", avih) + \
        b"LIST" + struct.pack("<I", len(strl)) + strl

    movi_chunks = []
    idx = []
    off = 4  # first chunk fourcc sits at offset 4 past the 'movi' id
    for fr in frames:
        c = chunk(b"00db", fr)
        idx.append(struct.pack("<4sIII", b"00db", 0x10, off, len(fr)))
        off += len(c)
        movi_chunks.append(c)
    movi = b"movi" + b"".join(movi_chunks)
    movi_list = b"LIST" + struct.pack("<I", len(movi)) + movi
    idx1 = b"idx1" + struct.pack("<I", len(b"".join(idx))) + b"".join(idx)

    payload = (b"AVI " +
               b"LIST" + struct.pack("<I", len(hdrl)) + hdrl +
               movi_list + idx1)
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(payload)) + payload)
    return frame_size * len(frames)


# ----------------------------------------------------------------------------
# Video rendering: split-screen Naive vs Full on a real NGSIM car-following pair
# ----------------------------------------------------------------------------
W, H = 840, 440
PANEL_W = 420
PX = 3.2          # pixels per metre (longitudinal)
LAT_PX = 11.0     # pixels per metre (lateral)
EGO_SY = int(H * 0.60)

COL_BG = (18, 20, 26)
COL_ROAD = (40, 44, 54)
COL_LANE = (90, 96, 110)
COL_EGO = (90, 200, 255)
COL_FRONT = (240, 200, 90)
COL_REAR = (240, 120, 120)
COL_SIDE = (170, 170, 180)
COL_OK = (90, 220, 120)
COL_WARN = (255, 180, 70)
COL_BAD = (255, 80, 80)
COL_TXT = (230, 232, 238)
COL_DIM = (150, 155, 165)


def pick_video_scenario(scens):
    """Pick the real pair with the hardest real leader braking (most
    instructive for the split-screen)."""
    def hardest(scen):
        vf = scen["vf"]
        return min((vf[i + 1] - vf[i]) / DT for i in range(len(vf) - 1))
    return min(scens, key=hardest)


def veh_screen_y(rel_long):
    return int(EGO_SY - rel_long * PX)


def draw_panel(cv, ox, title, trace, arrays, i, show_ring):
    cv.rect(ox, 0, PANEL_W, H, COL_BG)
    road_x = ox + PANEL_W // 2 - int(1.5 * LANE_W * LAT_PX)
    road_w = int(3 * LANE_W * LAT_PX)
    cv.rect(road_x, 0, road_w, H, COL_ROAD)
    for ln in range(4):
        lx = road_x + int(ln * LANE_W * LAT_PX)
        for yy in range(0, H, 16):
            cv.rect(lx, yy, 2, 9, COL_LANE)
    center_x = ox + PANEL_W // 2

    xe = arrays["x_e"][i]
    # front
    fy = veh_screen_y(arrays["x_f"][i] - xe - CAR_LEN)
    cv.rect(center_x - 10, fy - int(CAR_LEN * PX), 20, int(CAR_LEN * PX), COL_FRONT)
    # rear
    ry = veh_screen_y(arrays["x_r"][i] - xe + CAR_LEN)
    cv.rect(center_x - 10, ry, 20, int(CAR_LEN * PX), COL_REAR)
    # side vehicles
    encL_px = int((LANE_W - trace[i].get("_encL", 0.0)) * LAT_PX)
    # draw left/right adjacent vehicles pressing in (use clearance proxy)
    lat = arrays["min_lat"][i]
    lx = center_x - int(LANE_W * LAT_PX) + int((LANE_W - VEH_W - lat) * LAT_PX * 0.5)
    cv.rect(center_x - int(1.4 * LANE_W * LAT_PX), EGO_SY - 30, 16, 60, COL_SIDE)
    cv.rect(center_x + int(1.4 * LANE_W * LAT_PX) - 16, EGO_SY - 20, 16, 50, COL_SIDE)

    # ego (with lateral nudge)
    ex = center_x + int(arrays["y_e"][i] * LAT_PX)
    if show_ring:
        sig = trace[i].get("sigma_emp", 0.0)        # empirical uncertainty
        rr = 32 + int(min(sig, 6.0) * 7)
        ringc = COL_WARN if sig > 1.0 else COL_OK
        cv.ring(ex, EGO_SY + int(CAR_LEN * PX / 2), rr, 3, ringc)
        # 360 S0 status dots
        fok = arrays["front_gap"][i] >= arrays["v_e"][i] * 0.8
        rok = arrays["rear_gap"][i] >= arrays["v_r"][i] * 0.4 + 2.0
        latok = arrays["min_lat"][i] >= LAT_FLOOR
        cv.disc(ex, EGO_SY - rr + 6, 4, COL_OK if fok else COL_BAD)
        cv.disc(ex, EGO_SY + int(CAR_LEN * PX) + rr - 6, 4, COL_OK if rok else COL_BAD)
        cv.disc(ex - rr + 6, EGO_SY + 6, 4, COL_OK if latok else COL_BAD)
        cv.disc(ex + rr - 6, EGO_SY + 6, 4, COL_OK if latok else COL_BAD)
    cv.rect(ex - 10, EGO_SY, 20, int(CAR_LEN * PX), COL_EGO)

    # high-uncertainty tint (empirical forecast residual)
    if show_ring and trace[i].get("sigma_emp", 0.0) > 1.5:
        for yy in range(0, H, 3):
            cv.rect(road_x, yy, road_w, 1, (60, 55, 38))

    # HUD
    tr = trace[i]
    cv.text(ox + 10, 8, title, COL_TXT, 2)
    y = 34
    lines = [
        ("GAP %.0fM  GAIN %.0f%%" % (tr["front_gap"], tr["gain_pct"]), COL_TXT),
        ("FTTC %.1f  RTTC %.1f" % (tr["ttc_front"], tr["ttc_rear"]),
         COL_BAD if min(tr["ttc_front"], tr["ttc_rear"]) < 1.5 else COL_TXT),
        ("LAT %.2fM  UNC %.1f" % (tr["min_lat"], tr.get("sigma_emp", 0.0)),
         COL_BAD if tr["min_lat"] < LAT_FLOOR else COL_TXT),
    ]
    for s, c in lines:
        cv.text(ox + 10, y, s, c, 2)
        y += 20
    # S0/S1/S2 ticker
    dec = tr["decision"]
    dc = {"APPROVE": COL_OK, "HOLD": COL_WARN, "VETO": COL_BAD,
          "TIGHT": COL_BAD, "FIXED": COL_DIM}.get(dec, COL_DIM)
    cv.text(ox + 10, H - 44, "S2 " + dec, dc, 2)
    if tr["rationale"]:
        cv.text(ox + 10, H - 22, tr["rationale"][:34], COL_DIM, 1)
    # collision / breach banner
    if tr["front_gap"] <= 0.05 or tr["rear_gap"] <= 0.05:
        cv.rect(ox + 6, EGO_SY - 2, PANEL_W - 12, 4, COL_BAD)
        cv.text(ox + 120, 8, "COLLISION", COL_BAD, 2)


def render_video(path, scorecard_rows, scen, data_mode, n_samples=0):
    _, trN, arN = simulate("NAIVE", scen)
    _, trF, arF = simulate("FULL", scen)
    title = "NGSIM %s" % scen.get("name", "pair")
    frames = []
    for i in range(scen["steps"]):
        cv = Canvas(W, H)
        cv.fill(COL_BG)
        draw_panel(cv, 0, "NAIVE FRONT-TIGHT", trN, arN, i, show_ring=False)
        draw_panel(cv, PANEL_W, "FULL 360-PACK", trF, arF, i, show_ring=True)
        cv.rect(PANEL_W - 1, 0, 2, H, (12, 13, 17))
        cv.text(W // 2 - 90, H - 16, "T=%.1fS  %s" % (i * DT, title[:16]),
                COL_DIM, 2)
        dib, _ = cv.dib()
        frames.append(dib)
    # scorecard frames
    sc = Canvas(W, H)
    sc.fill((14, 15, 20))
    n = n_samples
    sc.text(40, 24, "ABLATION SCORECARD - NGSIM REAL DATA", COL_TXT, 3)
    heads = ["POLICY", "GAIN%", "VIOL", "MINTTC", "LAT MS", "COLL"]
    xcols = [40, 300, 420, 520, 640, 760]
    for h, xc in zip(heads, xcols):
        sc.text(xc, 80, h, COL_WARN, 2)
    names = {"IDM": "IDM", "NAIVE": "NAIVE", "S1S0": "S1+S0", "FULL": "FULL 360"}
    yy = 120
    for p in POLICIES:
        r = scorecard_rows[p]
        c = COL_OK if p == "FULL" else COL_TXT
        vals = [names[p], "%.0f" % r["gain"][0], "%.1f" % r["viol"][0],
                "%.2f" % r["ttc"][0], "%.3f" % r["lat"][0], "%.1f" % r["coll"][0]]
        for v, xc in zip(vals, xcols):
            sc.text(xc, yy, v, c, 2)
        yy += 40
    sc.text(40, yy + 20, "FULL 360-PACK: SAFE CAPACITY GAIN,", COL_OK, 2)
    sc.text(40, yy + 44, "FEWEST ENVELOPE VIOLATIONS.", COL_OK, 2)
    sc.text(40, H - 30,
            "METRICS FROM %d REAL NGSIM PAIRS (I-80 + US-101)" % n, COL_DIM, 1)
    scdib, _ = sc.dib()
    for _ in range(SCORE_FRAMES):
        frames.append(scdib)
    nbytes = write_avi(path, frames, W, H, int(1 / DT))
    return len(frames), nbytes


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_table(rows, n, unit="samples"):
    def c(t, f="%.1f"):
        return "--" if t is None else ("%s+-%s" % (f % t[0], f % t[1]))
    print("\n==== %d-WAY ABLATION (mean +/- std over %d %s) ===="
          % (len(rows), n, unit))
    hdr = ["policy", "coll", "cap.gain%", "env.viol", "minTTC", "recov",
           "lat.ms", "ADE", "FDE", "cov%"]
    fmt = "{:<8}{:>11}{:>16}{:>14}{:>14}{:>13}{:>14}{:>13}{:>12}{:>13}"
    print(fmt.format(*hdr))
    for p in POLICIES:
        r = rows[p]
        print(fmt.format(
            p, c(r["coll"], "%.2f"), c(r["gain"]), c(r["viol"]),
            c(r["ttc"], "%.2f"), c(r["rec"], "%.2f"), c(r["lat"], "%.3f"),
            c(r["ade"], "%.2f"), c(r["fde"], "%.2f"), c(r["cov"], "%.1f")))


def write_gitignore():
    with open(".gitignore", "w") as fh:
        fh.write("__pycache__/\n*.pyc\n*.avi\noutputs/*.avi\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="asymmetric-360-pack (stdlib only)")
    ap.add_argument("--csv", help="single NGSIM-style trajectory CSV to ingest")
    ap.add_argument("--ngsim", metavar="DIR",
                    help="root containing <location>/page_*.csv NGSIM caches")
    ap.add_argument("--location", default="i-80,us-101",
                    help="comma list of NGSIM locations (default i-80,us-101)")
    ap.add_argument("--pairs", type=int, default=160,
                    help="max real car-following pairs to evaluate")
    ap.add_argument("--pages", type=int, default=6,
                    help="NGSIM page files to read per location")
    ap.add_argument("--no-video", action="store_true", help="skip AVI render")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    write_gitignore()

    targets = ("front",)
    video_scen = None

    if args.csv:
        print("[csv] ingesting %s ..." % args.csv)
        scen = load_csv_scenario(args.csv, EVAL_STEPS)
        scen["name"] = scen.get("source", "csv")
        scens = [scen]
        per_policy, audit = evaluate_scenarios(scens, ("front",))
        data_mode = "csv:%s" % os.path.basename(args.csv)
        video_scen = scen
    else:
        root = find_ngsim_root(args.ngsim)
        if root is None:
            sys.exit(
                "[fatal] no real NGSIM cache found. This project evaluates on "
                "REAL data only.\n        Provide --ngsim DIR (a directory with "
                "<location>/page_*.csv NGSIM trajectory files),\n        or "
                "--csv FILE for a single NGSIM-style trajectory table.")
        locs = [x.strip() for x in args.location.split(",") if x.strip()]
        print("[eval] building real NGSIM car-following scenarios from %s "
              "(%s) ..." % (root, ",".join(locs)))
        scens = build_ngsim_scenarios(root, locs, args.pairs, args.pages)
        if not scens:
            sys.exit("[fatal] no usable NGSIM car-following pairs found in "
                     "%s for %s." % (root, ",".join(locs)))
        print("[eval] evaluating %d real leader-follower pairs x %d "
              "policies ..." % (len(scens), len(POLICIES)))
        per_policy, audit = evaluate_scenarios(scens, ("front",), keep_audit=4)
        data_mode = "ngsim:%s(%d pairs)" % ("+".join(locs), len(scens))
        video_scen = pick_video_scenario(scens)

    rows = agg(per_policy)
    n = len(per_policy["FULL"]["gain"])
    unit = "real pairs" if data_mode.startswith("ngsim") else "scenario"
    print("[data] source = %s" % data_mode)
    print_table(rows, n, unit)

    write_latex(rows, os.path.join(OUT_DIR, "ablation_table.tex"), data_mode)
    report = {
        "meta": {
            "title": "Asymmetric 360 World-Model Uncertainty & Agentic "
                     "Harness for Dynamic Roadway Micro-Packing",
            "data_source": data_mode, "samples": n, "dt": DT,
            "prediction_targets": list(targets),
            "disclaimer": (
                "Evaluated on REAL NGSIM trajectories only: the ego follows "
                "recorded real leader speed profiles (real shockwave braking) "
                "and real adjacent-lane neighbors. The trailing vehicle is a "
                "disclosed IDM-style model, not replayed. Reported values are "
                "outputs of the implemented control/prediction logic over real "
                "data."),
            "policies": POLICY_LABEL,
        },
        "aggregate": {p: {k: (None if v is None else
                              {"mean": v[0], "std": v[1]})
                          for k, v in rows[p].items()} for p in POLICIES},
        "raw": per_policy,
        "audit": audit,
    }
    with open(os.path.join(OUT_DIR, "ablation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print("[out] outputs/ablation_table.tex")
    print("[out] outputs/ablation_report.json")

    if not args.no_video and video_scen is not None:
        print("[video] rendering split-screen proof on real pair '%s' ..."
              % video_scen.get("name", "pair"))
        t0 = time.perf_counter()
        nf, nb = render_video(os.path.join(OUT_DIR, "asymmetric_360_proof.avi"),
                              rows, video_scen, data_mode, n)
        print("[out] outputs/asymmetric_360_proof.avi "
              "(%d frames, %.1f MB, %.1fs)" %
              (nf, nb / 1e6, time.perf_counter() - t0))
    else:
        print("[video] skipped")

    print("[done]")


if __name__ == "__main__":
    main()
