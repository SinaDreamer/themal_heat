"""
transient.py -- warm-up of the device after the SoC switches on.

Same finite-volume model as solver2d, plus heat storage in every cell:

    C_i dT_i/dt = Q_i - sum_j G_ij (T_i - T_j) - G_b,i (T_i - T_inf)

with C_i = rho*cp*volume. We march in time with backward (implicit) Euler:

    (C/dt + A) T^{n+1} = (C/dt) T^n + b

Implicit Euler is unconditionally stable, so we can take small steps while
the die heats up (seconds) and large ones while the enclosure creeps towards
steady state (tens of minutes). The film coefficients are refreshed whenever
the surface temperatures have moved by more than 0.5 K since the last update.

Two time scales show up: the die + TIM respond in seconds (tiny thermal
mass), the enclosure in ~10 minutes (most of the mass, behind the largest
resistances). That second time constant is what limits "burst" operation.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import device as dev
from solver2d import Model2D

# (step size s, run until t s): fine steps early, coarse later.
SCHEDULE = [(0.2, 20.0), (1.0, 120.0), (5.0, 1200.0), (15.0, 3 * 3600.0)]


@dataclass
class TransientResult:
    """Time histories from a warm-up run."""
    t: np.ndarray            # s
    T_junction: np.ndarray   # C, hottest die cell
    T_surface: np.ndarray    # C, hottest outer-surface point
    T_skin: np.ndarray       # C, hottest point on the skin-contact face
    energy_error: float      # (generated - stored - lost) / generated
    T_final: np.ndarray      # final temperature field


def run_warmup(variant: dev.Variant, schedule=SCHEDULE, refine=1.0) -> TransientResult:
    """Start uniformly at ambient, switch the SoC on at t=0, march to the end."""
    m = Model2D(variant, refine)
    g = m.grid
    C = (m.rho_cp * np.outer(g.dy, g.dx) * dev.DEPTH).ravel()   # J/K per cell
    T = np.full(g.n, m.t_amb)
    die = m.mask("SoC die").ravel()

    def surf(Tvec, h):
        return m.surface_temps(Tvec.reshape(g.ny, g.nx), h)

    ts = {f: np.full(g.nx if f in ("top", "bottom") else g.ny, m.t_amb)
          for f in ("top", "bottom", "left", "right")}
    h = m.h_coefficients(ts)
    ts_at_h = ts
    t_now, e_gen, e_out = 0.0, 0.0, 0.0
    E0 = np.sum(C * T)
    rec_t, rec_j, rec_s, rec_k = [0.0], [T[die].max()], [m.t_amb], [m.t_amb]
    lu, lu_key = None, None

    for dt, t_end in schedule:
        while t_now < t_end - 1e-9:
            # refresh h if the surfaces have drifted, then (re)factorise
            if max(np.max(np.abs(ts[f] - ts_at_h[f])) for f in ts) > 0.5:
                h, ts_at_h, lu = m.h_coefficients(ts), ts, None
            if lu is None or lu_key != dt:
                A, b = m.system(h)
                lu = spla.splu((A + sp.diags(C / dt)).tocsc())
                lu_key = dt
            T = lu.solve(C / dt * T + b)
            t_now += dt
            q_out = sum(m.heat_out(T.reshape(g.ny, g.nx), h).values())
            e_gen += variant.power * dt
            e_out += q_out * dt
            ts = surf(T, h)
            rec_t.append(t_now)
            rec_j.append(T[die].max())
            rec_s.append(max(v.max() for v in ts.values()))
            rec_k.append(ts["bottom"].max())

    stored = np.sum(C * T) - E0
    return TransientResult(np.array(rec_t), np.array(rec_j), np.array(rec_s),
                           np.array(rec_k), (e_gen - stored - e_out) / e_gen,
                           T.reshape(g.ny, g.nx))


def time_to(t, y, level):
    """First time (s) a rising history y(t) reaches level, or nan if never."""
    above = np.where(y >= level)[0]
    if above.size == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(t[0])
    return float(np.interp(level, [y[i - 1], y[i]], [t[i - 1], t[i]]))


def time_to_fraction(t, y, frac):
    """Time to reach a fraction (e.g. 0.95) of the final temperature rise."""
    return time_to(t, y, y[0] + frac * (y[-1] - y[0]))
