"""
sanity.py -- checks that the models are right before we trust any plot.

Each check returns (name, passed, detail). run_all.py writes them to
results/sanity_checks.txt and exits with an error if any fails.

What is checked and why:
  1. Energy balance (steady): heat generated == heat leaving all boundaries.
     Catches sign errors and missing boundary terms.
  2. Energy balance (transient): generated == stored + lost, and the warm-up
     ends at the steady-state answer.
  3. Grid convergence: the answer stops changing as the mesh is refined.
  4. 1D network vs 2D solver at shared nodes: two independent models agree.
  5. Physical plausibility: no temperature outside the physically possible
     range, sensible orderings, correlations used inside their validity
     range, monotonic trends, and a zero-power test.
"""

from dataclasses import replace

import numpy as np

import device as dev
import solver2d
from convection import rayleigh

NETWORK_TOL = 0.15     # 15 % of the temperature rise above ambient
ENERGY_TOL = 1e-6      # relative


def check_energy(results):
    """Steady energy balance for every variant."""
    out = []
    for k, r in results.items():
        err = r["m"]["energy_error"]
        out.append((f"steady energy balance, variant {k}", abs(err) < ENERGY_TOL,
                    f"(out - in)/in = {err:+.1e}  [Q_top {r['m']['Q_top']:.3f} + Q_ends "
                    f"{r['m']['Q_ends']:.3f} + Q_fins {r['m']['Q_fins']:.3f} + Q_body "
                    f"{r['m']['Q_skin']:.3f} W = {r['variant'].power:.2f} W]"))
    return out


def check_transient(runs, results):
    """Transient energy balance and convergence to the steady solution."""
    out = []
    for k, r in runs.items():
        out.append((f"transient energy balance, variant {k}", abs(r.energy_error) < ENERGY_TOL,
                    f"(generated - stored - lost)/generated = {r.energy_error:+.1e}"))
        d_j = r.T_junction[-1] - results[k]["m"]["T_junction"]
        d_s = r.T_surface[-1] - results[k]["m"]["T_surface_peak"]
        out.append((f"transient reaches steady state, variant {k}", max(abs(d_j), abs(d_s)) < 0.05,
                    f"after {r.t[-1] / 3600:.0f} h: junction off by {d_j:+.3f} K, surface by {d_s:+.3f} K"))
    return out


def check_grid(rows, richardson):
    """Mesh-independence of junction and surface temperature."""
    out = []
    for col, name in ((2, "junction"), (3, "peak surface")):
        f = rows[:, col]
        p, f_ex = richardson(*f[1:])
        out.append((f"grid convergence, {name}: finest two meshes agree", abs(f[-1] - f[-2]) < 0.1,
                    "values " + ", ".join(f"{v:.3f}" for v in f) + f" C on {', '.join(str(int(n)) for n in rows[:, 1])} cells"))
        out.append((f"grid convergence, {name}: default-mesh error", abs(f[1] - f_ex) < 0.5,
                    f"default mesh {f[1]:.3f} C vs Richardson extrapolation {f_ex:.3f} C "
                    f"(error {f[1] - f_ex:+.3f} K, observed order {p:.2f})"))
    return out


def check_network(nets, results):
    """1D network temperatures vs 2D solver at the shared nodes."""
    pairs = [("J", "T_die_mean", "junction"), ("To", "T_top_mean", "top surface"),
             ("P", "T_pcb_mean", "PCB"), ("Bo", "T_skin_mean", "skin-side surface")]
    out = []
    for k, net in nets.items():
        m = results[k]["m"]
        ta = results[k]["variant"].t_ambient
        errs, parts = [], []
        for node, key, label in pairs:
            rise2d = m[key] - ta
            e = (net.T[node] - m[key]) / rise2d
            errs.append(abs(e))
            parts.append(f"{label} {net.T[node]:.1f}/{m[key]:.1f} ({e:+.0%})")
        out.append((f"1D network vs 2D, variant {k}", max(errs) <= NETWORK_TOL,
                    "network/2D: " + ", ".join(parts)))
    return out


def check_physics(results, sweep):
    """Plausibility checks on the solutions."""
    out = []
    for k, r in results.items():
        sol, m = r["sol"], r["m"]
        lo = min(dev.T_BODY, sol.model.t_amb)
        die_max = sol.T[sol.model.mask("SoC die")].max()
        ok = sol.T.min() >= lo - 1e-9 and np.isclose(sol.T.max(), die_max)
        out.append((f"maximum principle, variant {k}", ok,
                    f"min {sol.T.min():.2f} C >= {lo:.0f} C, hottest cell is in the die ({die_max:.2f} C)"))
        ok = m["T_junction"] > m["T_pcb_mean"] > m["T_skin_mean"] > dev.T_BODY - 5 and \
            m["T_junction"] > m["T_surface_peak"] > sol.model.t_amb
        out.append((f"temperature ordering, variant {k}", ok,
                    f"Tj {m['T_junction']:.1f} > PCB {m['T_pcb_mean']:.1f} > skin side {m['T_skin_mean']:.1f}; "
                    f"Tj > surface peak {m['T_surface_peak']:.1f} > ambient"))
        ra = rayleigh(m["T_top_mean"], sol.model.t_amb,
                      dev.LENGTH * dev.DEPTH / (2 * (dev.LENGTH + dev.DEPTH)))
        out.append((f"natural-convection correlation in range, variant {k}",
                    1e4 <= ra <= 1e7 and 2 <= m["h_top_conv"] <= 15,
                    f"top Ra = {ra:.2e} (Lloyd-Moran valid 1e4-1e7), h_conv = {m['h_top_conv']:.2f} W/m2K"))
        e = r["exp"]
        ok = (abs(e["dL_um"] / e["dL_hand_um"] - 1) < 0.02
              and abs(e["rotation_urad"] / e["rotation_hand_urad"] - 1) < 0.02)
        out.append((f"bracket integral vs hand calc, variant {k}", ok,
                    f"dL {e['dL_um']:.2f} vs {e['dL_hand_um']:.2f} um; rotation {e['rotation_urad']:.3f} vs "
                    f"{e['rotation_hand_urad']:.3f} urad"))
    for name, arr in sweep.items():
        ok = np.all(np.diff(arr[:, 1]) <= 1e-6) and np.all(np.diff(arr[:, 2]) <= 1e-6)
        out.append((f"monotonic spreader sweep, {name}", ok,
                    f"junction {arr[0, 1]:.2f} -> {arr[-1, 1]:.2f} C, surface {arr[0, 2]:.2f} -> {arr[-1, 2]:.2f} C"))
    # zero power: the device can only sit between room and body temperature,
    # and the heat from the body must all leave through the outer surfaces.
    sol0 = solver2d.solve(replace(dev.VARIANTS[0], power=0.0))
    q = sol0.q_out
    ok = sol0.T.max() < dev.T_BODY and sol0.T.min() > dev.T_AMBIENT and \
        abs(sum(q.values())) < 1e-9
    out.append(("zero-power test", ok,
                f"T in [{sol0.T.min():.2f}, {sol0.T.max():.2f}] C, body supplies {-q['bottom']:.3f} W "
                f"which leaves through air faces ({q['top'] + q['left'] + q['right'] + q['fins']:.3f} W)"))
    return out


def report(checks):
    """Format the checks as text; returns (text, all_passed)."""
    lines = []
    for name, ok, detail in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    passed = all(ok for _, ok, _ in checks)
    n_ok = sum(ok for _, ok, _ in checks)
    lines.append(f"\n{n_ok}/{len(checks)} checks passed.")
    return "\n".join(lines), passed
