"""
network1d.py -- hand-calculation model: a thermal resistance network.

This is the model you would sketch on a whiteboard in an interview. Heat is
"current", temperature is "voltage", and every physical path is a resistor:

  conduction  R = t / (k A)          (through a layer of thickness t)
  convection  R = 1 / (h A)          (surface to air)
  contact     R = R''_c / A          (folded into the TIM's effective k)
  spreading   R_sp                   (extra resistance when heat enters a
                                      plate through a small footprint)
  fin         R = 1 / G_fin          (front/back/end walls as plastic fins)

Nodes (unknown temperatures):
  J   SoC junction (die mean)
  S   top-plate inner face (spreader / underside of the top wall)
  To  top outer surface (mean)
  P   PCB (mean)
  Bi  bottom wall inner face
  Bo  bottom (skin-side) outer surface (mean)
Fixed-temperature nodes: Amb (room air) and Body (core, behind the tissue).

Spreading resistance uses the exact Fourier-series solution for a strip
heat source on a layered plate cooled on the far face (Yovanovich-type
"flux channel" solution, here derived for a 2D strip so that it is directly
comparable with the 2D finite-volume model).

All geometry uses the same 2D-extruded convention as solver2d (strip
footprints across the full DEPTH, depth-homogenised die/TIM/bracket), so the
two models can be compared node by node.
"""

from dataclasses import dataclass

import numpy as np

import device as dev
from convection import h_horizontal_up, h_radiation, h_vertical


# ---------------------------------------------------------------------------
# Spreading resistance
# ---------------------------------------------------------------------------
def strip_spreading(c, d, layers, h_far, depth, n_terms=4000):
    """Thermal resistance of a strip source on a layered plate.

    A source of half-width c sits centred on a plate of half-width d. The plate
    is a stack of layers (thickness t, in-plane k_x, through-plane k_y) listed
    from the source side to the far side, and the far face is cooled with film
    coefficient h_far. Plate edges are adiabatic.

    The temperature is a Fourier cosine series. Mode n (lambda_n = n*pi/d) sees
    an effective "surface conductance" H_n at the source face, found by
    walking up through the layers from the far face:

        mu = lambda * sqrt(kx/ky),   K = ky*mu
        H_above = K * (K*tanh(mu t) + H_below) / (K + H_below*tanh(mu t))

    The mean (n = 0) mode is plain 1D conduction + convection. Summing the
    modes gives the source-average temperature rise:

        R_1D = (sum t/ky + 1/h) / (2 d D)
        R_sp = sum_n sin^2(lambda_n c) / (d c^2 lambda_n^2 H_n) / D

    Returns (R_1D, R_sp) in K/W.
    """
    r1d_area = sum(t / ky for t, _, ky in layers) + 1.0 / h_far
    r_1d = r1d_area / (2 * d * depth)
    lam = np.arange(1, n_terms + 1) * np.pi / d
    H = np.full_like(lam, h_far)
    for t, kx, ky in reversed(layers):
        mu = lam * np.sqrt(kx / ky)
        K = ky * mu
        th = np.tanh(mu * t)
        H = K * (K * th + H) / (K + H * th)
    r_sp = np.sum(np.sin(lam * c) ** 2 / (d * c ** 2 * lam ** 2 * H)) / depth
    return r_1d, r_sp


def fin_conductance(h, k, t, length, width):
    """Conductance of a straight fin with adiabatic tip [Incropera eq. 3.81].

    G = sqrt(h P k A_c) * tanh(m L), with m = sqrt(h / (k t)) for a plate fin of
    thickness t cooled on one face, perimeter ~ width, cross-section t*width.
    """
    m = np.sqrt(h / (k * t))
    return np.sqrt(h * k * t) * width * np.tanh(m * length)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
@dataclass
class Resistor:
    """One labelled resistor between two nodes."""
    name: str
    a: str
    b: str
    R: float           # K/W
    detail: str = ""   # breakdown shown on the diagram


@dataclass
class NetworkResult:
    """Node temperatures, resistor list and heat flows of a solved network."""
    T: dict
    resistors: list
    flows: dict
    h: dict


def _solve_nodes(resistors, fixed, sources):
    """Nodal analysis: sum of (T_a - T_b)/R at each free node = source."""
    free = sorted({n for r in resistors for n in (r.a, r.b)} - set(fixed))
    idx = {n: i for i, n in enumerate(free)}
    A = np.zeros((len(free), len(free)))
    b = np.zeros(len(free))
    for r in resistors:
        g = 1.0 / r.R
        for p, q in ((r.a, r.b), (r.b, r.a)):
            if p in idx:
                A[idx[p], idx[p]] += g
                if q in idx:
                    A[idx[p], idx[q]] -= g
                else:
                    b[idx[p]] += g * fixed[q]
    for n, q in sources.items():
        b[idx[n]] += q
    T = dict(fixed)
    T.update(zip(free, np.linalg.solve(A, b)))
    return T


def build_resistors(v: dev.Variant, h):
    """Assemble the resistor list for a variant, given film coefficients h."""
    lay = dev.build_layout(v)
    y = lay["y_levels"]
    rect = {r.name: r for r in lay["rects"]}
    D, L = dev.DEPTH, dev.LENGTH
    w = dev.DIE_WIDTH
    die, tim, pcb = rect["SoC die"].material, rect["TIM"].material, dev.PCB
    shell = v.top_shell or dev.PC
    A_top = L * D
    A_die = w * D

    # ---- top path: J -> S (die half + TIM + spreading in the top plate) ----
    r_die = (dev.DIE_THICK / 2) / (die.k_y * A_die)
    r_tim = v.tim.thickness / (tim.k_y * A_die)
    layers = []
    if v.spreader is not None:
        layers.append((v.spreader_thick, v.spreader.k_x, v.spreader.k_y))
    layers.append((dev.T_WALL, shell.k_x, shell.k_y))
    _, r_sp_top = strip_spreading(w / 2, L / 2, layers, h["top"], D)

    # ---- S -> To: 1D conduction through spreader + wall ----
    r_plate = dev.T_WALL / (shell.k_y * A_top)
    if v.spreader is not None:
        r_plate += v.spreader_thick / (v.spreader.k_y * (dev.SPREADER_X1 - dev.SPREADER_X0) * D)

    # ---- J -> P: die half + PCB (half thickness + spreading) ----
    A_pcb = (dev.PCB_X1 - dev.PCB_X0) * D
    gap_top = y["top_inner"] - v.spreader_thick - y["die_bot"]  # air between PCB and plate
    gap_bot = y["pcb_bot"] - y["skin_spr_top"]                  # PCB underside to bottom wall
    r_up_area = gap_top / dev.AIR.k_y + dev.T_WALL / shell.k_y + 1 / h["top"]
    r_dn_area = gap_bot / dev.AIR.k_y + dev.T_WALL / dev.PC.k_y + 1 / dev.H_SKIN
    h_eff_pcb = 1 / r_up_area + 1 / r_dn_area
    _, r_sp_pcb = strip_spreading(w / 2, (dev.PCB_X1 - dev.PCB_X0) / 2,
                                  [(dev.PCB_THICK, pcb.k_x, pcb.k_y)], h_eff_pcb, D)
    r_pcb_half = (dev.PCB_THICK / 2) / (pcb.k_y * A_pcb)

    # ---- P -> S: conduction through the air pocket beside the die ----
    r_air_top = gap_top / (dev.AIR.k_y * (A_pcb - A_die)) + r_pcb_half

    # ---- P -> Bi: below the PCB. Under the bracket: pad + bracket + air/bridge;
    #      elsewhere: one air gap. The two columns are in parallel. ----
    A_brk = dev.BRACKET_LENGTH * D
    A_rest = A_pcb - A_brk
    brk, pad = rect["bracket"].material, rect["bracket pad"].material
    below_brk = y["brk_bot"] - y["skin_spr_top"]
    filler = rect["skin bridge"].material if v.skin_bridge else dev.AIR
    r_col_brk = (dev.BRACKET_PAD_THICK / pad.k_y + dev.BRACKET_THICK / brk.k_y
                 + below_brk / filler.k_y) / A_brk
    r_col_rest = (gap_bot / dev.AIR.k_y) / A_rest
    r_below = 1 / (1 / r_col_brk + 1 / r_col_rest) + r_pcb_half

    # ---- bottom wall and skin ----
    r_bot_wall = dev.T_WALL / (dev.PC.k_y * A_top)
    r_skin = 1 / (dev.H_SKIN * A_top)

    # ---- outer losses: top face, plus end walls and front/back walls as fins.
    #      Each wall is split at mid-height: the top half is fed by the top
    #      wall, the bottom half by the bottom wall. ----
    r_top_conv = 1 / (h["top"] * A_top)
    g_fin_half = (fin_conductance(h["side"], dev.PC.k_x, dev.T_WALL, dev.HEIGHT / 2, 2 * D)
                  + fin_conductance(h["side"], dev.PC.k_x, dev.T_WALL, dev.HEIGHT / 2, 2 * L))

    return [
        Resistor("R_die+TIM+spread", "J", "S", r_die + r_tim + r_sp_top,
                 f"die {r_die:.2f} + TIM {r_tim:.2f} + spread {r_sp_top:.2f}"),
        Resistor("R_shell", "S", "To", r_plate, "spreader + top wall (1D)"),
        Resistor("R_conv,top", "To", "Amb", r_top_conv, "convection + radiation"),
        Resistor("R_fins,top", "To", "Amb", 1 / g_fin_half, "end + front/back walls (fins)"),
        Resistor("R_die+PCB", "J", "P", r_die + r_sp_pcb + r_pcb_half,
                 f"die {r_die:.2f} + spread {r_sp_pcb:.2f} + PCB {r_pcb_half:.2f}"),
        Resistor("R_air,top", "P", "S", r_air_top, "air pocket beside the die"),
        Resistor("R_gap,bottom", "P", "Bi", r_below, "pad/bracket || air gap"),
        Resistor("R_wall,bottom", "Bi", "Bo", r_bot_wall, "bottom wall"),
        Resistor("R_skin", "Bo", "Body", r_skin, "tissue + strap contact"),
        Resistor("R_fins,bottom", "Bo", "Amb", 1 / g_fin_half, "end + front/back walls (fins)"),
    ]


def solve_network(v: dev.Variant, tol=1e-4, max_iter=50) -> NetworkResult:
    """Solve the network, iterating the temperature-dependent h values.

    The top-face h is area-averaged over the vented and sealed regions.
    """
    ta = v.t_ambient
    t_top = t_side = ta + 15
    vent = dev.vent_region(v)
    vent_frac = 0.0 if vent is None else (vent[1] - vent[0]) / dev.LENGTH
    for _ in range(max_iter):
        hc = h_horizontal_up(t_top, ta, dev.LENGTH * dev.DEPTH, 2 * (dev.LENGTH + dev.DEPTH))
        hc *= 1 + (dev.VENT_FACTOR - 1) * vent_frac
        h = {"top": hc + float(h_radiation(t_top, ta)),
             "side": h_vertical(t_side, ta, dev.HEIGHT) + float(h_radiation(t_side, ta))}
        res = build_resistors(v, h)
        T = _solve_nodes(res, {"Amb": ta, "Body": dev.T_BODY}, {"J": v.power})
        new_side = 0.5 * (T["To"] + T["Bo"])
        if abs(T["To"] - t_top) < tol and abs(new_side - t_side) < tol:
            break
        t_top, t_side = T["To"], new_side
    flows = {r.name: (T[r.a] - T[r.b]) / r.R for r in res}
    return NetworkResult(T, res, flows, h)
