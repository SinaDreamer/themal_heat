"""
analysis.py -- studies built on top of the solvers.

  * run_variants      : steady state + bracket deformation for every variant
  * sustainable_power : the highest SoC power that keeps every outer surface
                        at or below the 43 C limit (the device's thermal budget)
  * spreader_sweep    : junction / surface temperature vs spreader thickness
  * grid_convergence  : the same case on successively finer meshes
  * gap_closers       : what it would take to run 3.5 W continuously
"""

from dataclasses import replace

import numpy as np

import device as dev
import expansion
import solver2d


def run_variants(variants=None):
    """Solve every variant; returns {key: dict(variant, sol, m, exp, p_max)}."""
    out = {}
    for v in variants or dev.VARIANTS:
        sol = solver2d.solve(v)
        out[v.key] = dict(variant=v, sol=sol, m=sol.metrics,
                          exp=expansion.bracket_deformation(sol),
                          p_max=sustainable_power(v),
                          mass=dev.added_mass_g(v))
    return out


def sustainable_power(v: dev.Variant, limit=dev.SKIN_LIMIT, tol=0.005):
    """Highest continuous power (W) with peak surface temperature <= limit.

    Temperatures are almost linear in power (only h varies, weakly), so a
    secant iteration on f(P) = T_peak(P) - limit converges in 3-4 solves.
    """
    def f(p):
        return solver2d.solve(replace(v, power=p)).metrics["T_surface_peak"] - limit

    p0, p1 = 2.0, dev.POWER
    f0, f1 = f(p0), f(p1)
    for _ in range(20):
        p2 = p1 - f1 * (p1 - p0) / (f1 - f0)
        if abs(p2 - p1) < tol:
            return p2
        p0, f0, p1, f1 = p1, f1, p2, f(p2)
    raise RuntimeError("sustainable power search did not converge")


SWEEP_THICKNESS_MM = [0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def spreader_sweep(base: dev.Variant = None, materials=(dev.COPPER, dev.PGS),
                   thicknesses_mm=SWEEP_THICKNESS_MM):
    """Junction and peak-surface temperature vs spreader thickness.

    Thick graphite is modelled as a stack of sheets with the same in-plane k
    (real thick PGS grades have lower k, so the thick end is optimistic).
    """
    base = base or dev.VARIANTS[0]
    res = {}
    for mat in materials:
        rows = []
        for t in thicknesses_mm:
            v = dev.with_spreader(base, mat, t * dev.MM)
            m = solver2d.solve(v).metrics
            rows.append((t, m["T_junction"], m["T_surface_peak"], dev.added_mass_g(v)))
        res[mat.name] = np.array(rows)
    return res


def grid_convergence(v: dev.Variant, refines=(0.5, 1.0, 2.0, 4.0)):
    """Solve on meshes refined by each factor; returns rows of
    (refine, n_cells, T_junction, T_surface_peak, bracket_dT)."""
    rows = []
    for r in refines:
        m = solver2d.solve(v, refine=r).metrics
        rows.append((r, m["n_cells"], m["T_junction"], m["T_surface_peak"], m["bracket_dT"]))
    return np.array(rows)


def richardson(f_coarse, f_mid, f_fine, ratio=2.0):
    """Observed order of accuracy and extrapolated value from three meshes."""
    p = np.log(abs(f_coarse - f_mid) / abs(f_mid - f_fine)) / np.log(ratio)
    f_exact = f_fine + (f_fine - f_mid) / (ratio ** p - 1)
    return p, f_exact


def gap_closers():
    """Options beyond the recommended design, each evaluated at 3.5 W.

    These are the levers left once spreading has flattened the hotspot:
    more surface h (vents), a second heat path (into the body), a more
    conductive shell -- and a reminder of how a hot room changes the picture.
    """
    f = dev.variant_by_key(dev.RECOMMENDED_KEY)
    skin_path = dict(skin_spreader=dev.PGS, skin_spreader_thick=0.1 * dev.MM, skin_bridge=True)
    options = [
        ("Recommended (f), 25 C room", f),
        ("f + thin high-k TIM", replace(f, tim=dev.TIM_GEL)),
        ("f + vents over SoC", replace(f, vented=True)),
        ("f + skin-side path (PGS + gap filler)", replace(f, **skin_path)),
        ("f + conductive top shell (2 W/mK)", replace(f, top_shell=dev.PC_CONDUCTIVE)),
        ("f + vents + skin path", replace(f, vented=True, **skin_path)),
        ("Recommended (f), 35 C room", replace(f, t_ambient=35.0)),
    ]
    rows = []
    for name, v in options:
        m = solver2d.solve(v).metrics
        rows.append(dict(option=name, T_junction=m["T_junction"],
                         T_top_peak=m["T_top_peak"], T_skin_peak=m["T_skin_peak"],
                         bracket_dT=m["bracket_dT"], mass=dev.added_mass_g(v),
                         p_max=sustainable_power(v)))
    return rows
