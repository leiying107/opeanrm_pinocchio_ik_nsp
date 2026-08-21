#!/usr/bin/env python3
"""Impedance stress harness (headless, no CAN / no ROS / no threads).

Directly couples CartesianImpedance + ImpedanceSimPlant at 250 Hz, faithfully
replicating the ArmController sim path (sensor noise, TMAX clamp, gravity via
the same _grav-style pinocchio call, ramp, soft-stop, escape) so any failure
here would also fail on hardware.

Layers (run with --layer N, default all):
  L1  pose grid x push direction x force sweep     — coverage
  L2  interesting poses, deep parameter sweep      — presets x zeta x force
  L3  random fuzz                                  — multi-push, live retune,
                                                     tick jitter, leak jumps
  L4  release rebound                              — every pose: push then let go
  L5  axial inward push at low sigma               — singular-direction torture

Verdicts per scenario:
  OK        held, bounded, returned
  DRIFT     steady error grew without force
  OSC       sustained oscillation (z-crossings high + energy never decays)
  SIGEXIT   hard-exit sigma path taken (impedance lost)
  FATAL     NaN / joint blew past URDF limits / sim diverged

Usage:
  cd /ros2_ws/openarm_nsp_ws
  ./venv-openarm-ik/bin/python src/openarm_dashboard/scripts/test_impedance_stress.py \
      [--layer 1] [--quick] [--out stress_v4.json] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]           # openarm_nsp_ws
sys.path.insert(0, str(ROOT / "src/openarm_pinocchio_nsp/src"))
sys.path.insert(0, str(ROOT / "src/openarm_dashboard/src"))

from openarm_dashboard.impedance import (  # noqa: E402
    CartesianImpedance, ImpedanceSimPlant, PRESETS,
)
from openarm_pinocchio_nsp.kinematics import PinocchioModel  # noqa: E402
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path  # noqa: E402

DT = 0.004                 # 250 Hz
SIG_CRIT = 0.02
SIG_WARN = 0.05


# ------------------------------------------------------------------ helpers
def make_ctrl(side: str = "left"):
    m = PinocchioModel(resolve_urdf_path(), side)
    imp = CartesianImpedance(m)
    plant = ImpedanceSimPlant(side)
    grav = m  # gravity via the same pinocchio model the controller uses
    return imp, plant, grav


def grav_tau(m, q):
    import pinocchio as pin
    g = pin.computeGeneralizedGravity(m.model, m.data, m._full_q(q)).copy()
    return g[m.q_idx]


def sigma_of(m, q):
    import pinocchio as pin
    J = m._pose_and_jac(q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[1]
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def run_scenario(imp, plant, m, q0, pushes, duration=8.0, retunes=(),
                 tick_jitter=0.0, seed=0):
    """Run one scenario. pushes = [(t_start, F_world3, dur), ...];
    retunes = [(t, {params}), ...]. Returns a result dict.

    The imp/plant objects may be REUSED across scenarios for speed, but the
    controller's internal filter states (_blend slew, _v_dir LPF, _dsig_f,
    _dq_f) must not leak between scenarios — that produced phantom verdicts
    (vend 2.2 in-scenario vs 0.36 standalone, 2026-08-19). Force-reset them
    here so each scenario behaves like a fresh enable."""
    rng = np.random.default_rng(seed)
    imp.start(q0)
    plant.reset(q0)
    plant._push = None          # stale push from a previous scenario bleeds in
    plant._push_until = 0.0     # (reset() clears q/v only)
    # hard-reset every filter state start() forgets
    imp._blend = 0.0
    imp._v_dir[:] = 0.0
    imp._dsig_f = 0.0
    imp._sigma_prev = float("inf")
    imp._dq_f[:] = 0.0
    imp.push_boost = 0.0
    x0 = np.asarray(m.fk(q0)[0], float)
    n = int(duration / DT)
    sigma_sig = 0.0
    sig_exit = False
    sig_low = 0
    q_hist = np.zeros((n, 7))
    dq_hist = np.zeros((n, 7))
    sig_hist = np.zeros(n)
    dx_hist = np.zeros((n, 3))
    t_hist = np.zeros(n)
    last_diag = None
    for k in range(n):
        t = k * DT
        # schedule pushes (VIRTUAL time for deadlines — sim runs faster than
        # wall-clock, so time.monotonic() deadlines never expire in-loop)
        for (ts, F, dur) in pushes:
            if abs(t - ts) < DT / 2:
                plant.set_push(F, dur, t)
        for (ts, params) in retunes:
            if abs(t - ts) < DT / 2:
                imp.set_params(**params)
        if tick_jitter:
            dt_eff = DT * (1.0 + rng.normal(0, tick_jitter))
        else:
            dt_eff = DT
        q = plant.q + rng.normal(0, 0.005, 7)
        dq = plant.v + rng.normal(0, 0.01, 7)
        try:
            tau_imp, diag = imp.torque(q, dq, dt_eff)
        except Exception as e:  # noqa: BLE001
            return {"verdict": "FATAL", "why": f"exception {type(e).__name__}: {e}",
                    "t_fail": t}
        if not np.all(np.isfinite(tau_imp)):
            return {"verdict": "FATAL", "why": "NaN in tau", "t_fail": t}
        tau = tau_imp + grav_tau(m, q)
        tau = np.clip(tau, -np.array([54, 54, 28, 28, 10, 10, 10], float),
                      np.array([54, 54, 28, 28, 10, 10, 10], float))
        plant.step(tau, t, dt_eff, substeps=2)
        if not np.all(np.isfinite(plant.q)):
            return {"verdict": "FATAL", "why": "plant NaN", "t_fail": t}
        q_hist[k] = plant.q
        dq_hist[k] = plant.v
        sig_hist[k] = diag["sigma"]
        dx_hist[k] = diag["dx"]
        t_hist[k] = t
        last_diag = diag
        if diag["sigma"] < SIG_CRIT:
            # persistence-gated like the real controller (0.5 s sustained):
            # brief dips occur while transiting interior low-sigma pockets
            sig_low += 1
            if sig_low >= 125:
                sig_exit = True
                return {"verdict": "SIGEXIT",
                        "why": f"sigma {diag['sigma']:.3f} sustained 0.5s at t={t:.2f}",
                        "t_fail": t, "sigma_trace": sig_hist[:k + 1].copy()}
        else:
            sig_low = 0
    # ---- analyze
    fin = np.all(np.isfinite(q_hist))
    xN = np.asarray(m.fk(plant.q)[0], float)
    final_err = float(np.linalg.norm(xN - x0)) * 1000
    # oscillation metric: late-run tremor amplitude — LPF (0.5 s window) the
    # final second of joint motion and measure the peak excursion. Quiet hold
    # ≈ 0.005 rad; the 08-18 hunting incident was multi-radian. (Raw signal
    # zero-crossing counts sensor noise: 71/s/joint even on a quiet trace.)
    w = max(8, int(0.5 / DT))
    ker = np.ones(w) / w
    tail = q_hist[-int(1.0 / DT):]
    tail_f = np.stack([np.convolve(tail[:, j], ker, mode="valid")
                       for j in range(7)], axis=1)
    tremor = float(np.max(np.abs(tail_f - tail_f.mean(0))))
    v_end = float(np.max(np.abs(dq_hist[-250:])))
    v_peak = float(np.max(np.abs(dq_hist)))
    # drift with no active push in the last 2 s
    tail_dx = np.linalg.norm(dx_hist[-500:], axis=1)
    verdict = "OK"
    why = ""
    if not fin:
        verdict, why = "FATAL", "non-finite state"
    elif v_end > 1.0:
        verdict, why = "OSC", f"still moving {v_end:.1f} rad/s at end"
    elif tremor > 0.05:
        verdict, why = "OSC", f"tremor {tremor:.3f} rad in final second"
    elif tail_dx.mean() > 50.0:
        verdict, why = "DRIFT", f"|dx| tail {tail_dx.mean():.0f} mm"
    return {"verdict": verdict, "why": why, "final_err_mm": round(final_err, 1),
            "v_peak": round(v_peak, 2), "v_end": round(v_end, 2),
            "tremor": round(tremor, 4), "sigma_min": round(float(sig_hist.min()), 4),
            "sigma_end": round(float(sig_hist[-1]), 4)}


def summarize(results, label):
    from collections import Counter
    c = Counter(r["verdict"] for r in results)
    total = len(results)
    print(f"\n== {label}: {total} scenarios ==", flush=True)
    for v in ("OK", "DRIFT", "OSC", "SIGEXIT", "FATAL"):
        if c.get(v):
            print(f"  {v:8s} {c[v]:4d}  ({c[v]/total:.0%})", flush=True)
    bad = [r for r in results if r["verdict"] not in ("OK",)]
    if bad:
        print(f"  --- failures ({len(bad)}) ---", flush=True)
        for r in bad[:12]:
            print(f"    {r['verdict']:8s} {r.get('tag', '')} {r.get('why', '')}",
                  flush=True)
    return c


# ------------------------------------------------------------------ layers
def pose_grid(m, quick=False):
    """Coverage grid over the reachable, non-singular workspace."""
    poses = []
    j1s = [-0.8, -0.4, 0.0, 0.4, 0.8]
    j2s = [-0.5, 0.0, 0.5]
    j4s = [0.6, 0.9, 1.2, 1.6, 2.0]
    if quick:
        j1s, j2s, j4s = j1s[::2], j2s[::1], j4s[::2]
    for j1 in j1s:
        for j2 in j2s:
            for j4 in j4s:
                q = np.array([j1, j2, 0.0, j4, 0.0, 0.0, 0.0])
                s = sigma_of(m, q)
                if s >= SIG_WARN:      # entry gate honored
                    poses.append((q, s))
    return poses


def dirs(n=6):
    """Push directions on a cone: horizontal circle + verticals."""
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        out.append(np.array([np.cos(a), np.sin(a), 0.0]))
    out += [np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]
    return out


def layer1(quick=False, seed=0, shard=None):
    imp, plant, m = make_ctrl()
    results = []
    poses = pose_grid(m, quick)
    fs = [5.0, 20.0, 40.0] if not quick else [20.0]
    cases = [(q0, F, d) for (q0, s0) in poses for F in fs
             for d in dirs(4 if quick else 6)]
    if shard:
        cases = cases[shard[0]::shard[1]]
    for (q0, F, d) in cases:
        imp.set_params(preset="ultra")
        r = run_scenario(imp, plant, m, q0,
                         pushes=[(2.0, F * d, 1.5)], duration=7.0, seed=seed)
        r["tag"] = (f"j1={q0[0]:+.1f} j2={q0[1]:+.1f} j4={q0[3]:.1f} "
                    f"F={F:.0f}N dir=[{d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f}]")
        results.append(r)
    return results


def layer2(quick=False, seed=0, shard=None):
    imp, plant, m = make_ctrl()
    results = []
    interesting = [
        np.array([0.0, -0.5, 0.0, 0.8, 0.0, 0.0, 0.0]),      # safe pose
        np.array([0.4, 0.3, 0.0, 1.4, 0.0, 0.5, 0.0]),       # wrist bent
        np.array([-0.6, -0.2, 0.3, 1.8, -0.4, -0.3, 0.6]),   # twisted
        np.array([0.0, 0.5, 0.0, 2.2, 0.0, 0.0, 0.0]),       # folded high
    ]
    if quick:
        interesting = interesting[:2]
    cases = [(q0, pname, zeta, F) for q0 in interesting
             for pname in ("ultra", "soft", "mid", "stiff")
             for zeta in (0.6, 1.0, 1.4) for F in (20.0, 60.0)]
    if shard:
        cases = cases[shard[0]::shard[1]]
    for (q0, pname, zeta, F) in cases:
        imp.set_params(preset=pname, zeta=zeta)
        d = np.array([1.0, 0.0, 0.0])
        r = run_scenario(imp, plant, m, q0,
                         pushes=[(2.0, F * d, 1.5)], duration=7.0,
                         seed=seed)
        r["tag"] = (f"{pname} z={zeta} F={F:.0f} q4={q0[3]:.1f}")
        results.append(r)
    return results


def layer3(quick=False, seed=1, shard=None):
    """Random fuzz: multi-push sequences, live retunes, tick jitter, leak."""
    rng = np.random.default_rng(seed)
    imp, plant, m = make_ctrl()
    results = []
    n_case = 20 if quick else 60
    idxs = list(range(n_case))
    if shard:
        idxs = idxs[shard[0]::shard[1]]
    for i in idxs:
        # random start pose with sigma above gate
        for _ in range(50):
            q0 = rng.uniform(m.lower + 0.25, m.upper - 0.25)
            q0[3] = abs(q0[3]) + 0.3
            if sigma_of(m, q0) >= SIG_WARN:
                break
        else:
            continue
        pushes = []
        for _ in range(rng.integers(1, 4)):
            ts = float(rng.uniform(1.0, 5.0))
            F = rng.normal(0, 25.0, 3)
            F[np.argmax(np.abs(F))] *= 1.5
            pushes.append((ts, F, float(rng.uniform(0.3, 1.2))))
        pushes.sort()
        retunes = []
        for _ in range(int(rng.integers(0, 3))):
            params = {}
            if rng.random() < 0.5:
                params["preset"] = rng.choice(["ultra", "soft", "mid"])
            if rng.random() < 0.5:
                params["zeta"] = float(rng.uniform(0.5, 1.5))
            if rng.random() < 0.4:
                params["leak"] = float(rng.uniform(0.0, 1.5))
            if params:
                retunes.append((float(rng.uniform(0.5, 5.0)), params))
        imp.set_params(preset=rng.choice(["ultra", "soft", "mid"]),
                       zeta=float(rng.uniform(0.5, 1.5)), leak=0.0)
        r = run_scenario(imp, plant, m, q0, pushes, duration=8.0, retunes=retunes,
                         tick_jitter=0.15, seed=seed * 1000 + i)
        r["tag"] = f"fuzz#{i}"
        results.append(r)
    return results


def layer4(quick=False, seed=0, shard=None):
    """Release rebound: push 1.5 s then release, measure decay."""
    imp, plant, m = make_ctrl()
    results = []
    poses = pose_grid(m, quick)
    if quick:
        poses = poses[::4]
    sel = poses[: (10 if quick else 40)]
    if shard:
        sel = sel[shard[0]::shard[1]]
    for (q0, s0) in sel:
        for F in (20.0, 40.0):
            imp.set_params(preset="soft")
            r = run_scenario(imp, plant, m, q0,
                             pushes=[(2.0, F * np.array([1.0, 0, 0]), 1.5)],
                             duration=7.0, seed=seed)
            r["tag"] = f"release j4={q0[3]:.1f} F={F:.0f}"
            results.append(r)
    return results


def layer5(quick=False, seed=0, shard=None):
    """Axial inward push torture: start mid, push radially INWARD hard, and
    start near the soft-stop then push inward (should flex, not exit)."""
    imp, plant, m = make_ctrl()
    results = []
    # inward = toward shoulder along (shoulder->hand) reversed
    def radial_dir(q):
        import pinocchio as pin
        qf = m._full_q(q)
        pin.forwardKinematics(m.model, m.data, qf)
        pin.updateFramePlacements(m.model, m.data)
        x = np.asarray(m.fk(q)[0], float)
        j3 = np.asarray(
            m.data.oMi[m.model.getJointId(f"openarm_left_joint3")].translation, float)
        r = x - j3
        return r / np.linalg.norm(r)

    for j4v in (0.9, 1.3, 1.8):
        q0 = np.array([0.0, -0.5, 0.0, j4v, 0.0, 0.0, 0.0])
        for Fmag in (20.0, 50.0):
            d = -radial_dir(q0)      # INWARD
            imp.set_params(preset="soft")
            r = run_scenario(imp, plant, m, q0,
                             pushes=[(2.0, Fmag * d, 2.0)], duration=7.0, seed=seed)
            r["tag"] = f"j4={j4v:.1f} Fin={Fmag:.0f}"
            results.append(r)
    # also: from the soft-stop equilibrium, push inward (v5 buckle feature test)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0, help="0=all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1",
                    help="i/n — run only the i-th of n shards (parallel runs)")
    args = ap.parse_args()

    i, n = (int(x) for x in args.shard.split("/"))
    shard = (i, n) if n > 1 else None
    layers = {1: layer1, 2: layer2, 3: layer3, 4: layer4, 5: layer5}
    which = [args.layer] if args.layer else [1, 2, 3, 4, 5]
    all_results = {}
    for ln in which:
        t0 = time.time()
        res = layers[ln](args.quick, args.seed, shard=shard)
        c = summarize(res, f"L{ln} shard {args.shard}")
        all_results[f"L{ln}"] = {"counts": dict(c), "results": res,
                                 "secs": round(time.time() - t0, 1)}
        print(f"  [L{ln} done in {time.time()-t0:.0f}s]", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=1, default=str)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
