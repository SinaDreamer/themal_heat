"""
solver2d.py -- 2D finite-volume conduction solver for the device cross-section.

Method (the "finite-difference" model of the brief, written in its
finite-volume form, which is what makes material jumps behave):

  * The cross-section is split into rectangular cells on a non-uniform grid.
    Every material interface is a grid line, so each cell holds one material.
  * Each cell is a node. Neighbouring nodes are joined by a conductance
        G = (face area) / (dx_1/(2 k_1) + dx_2/(2 k_2))
    i.e. two half-cells in series -- the harmonic mean of k. This is exactly
    the thermal-resistance-network idea, just with thousands of nodes.
  * Outer faces connect to the ambient / body through the half-cell plus the
    film resistance 1/h. Heat generation is added to the die cells.
  * Energy conservation at every node gives a sparse linear system A T = b.
  * h depends on the surface temperature (natural convection, radiation), so
    we iterate: solve -> update h -> solve, until surface temps stop moving.

Units: SI. Conductances are multiplied by the device DEPTH so that heat flows
are in watts for the real (extruded) device.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import device as dev
from convection import h_horizontal_up, h_radiation, h_vertical


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def _edges(breaks, max_size, min_cells):
    """Grid-line coordinates that hit every break point.

    Each interval between consecutive breaks is split into equal cells no
    larger than max_size, and at least min_cells (so a 0.1 mm graphite layer
    still gets more than one cell through its thickness).
    """
    b = np.unique(np.round(np.asarray(breaks, dtype=float), 9))
    edges = [b[0]]
    for a, c in zip(b[:-1], b[1:]):
        n = max(min_cells, int(np.ceil((c - a) / max_size - 1e-9)))
        edges.extend(a + (c - a) * np.arange(1, n + 1) / n)
    return np.array(edges)


@dataclass
class Grid:
    """Cell edges, centres and sizes of the non-uniform rectilinear mesh."""
    xe: np.ndarray
    ye: np.ndarray

    def __post_init__(self):
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        self.yc = 0.5 * (self.ye[1:] + self.ye[:-1])
        self.dx = np.diff(self.xe)
        self.dy = np.diff(self.ye)
        self.nx = self.xc.size
        self.ny = self.yc.size

    @property
    def n(self) -> int:
        """Total number of cells (unknowns)."""
        return self.nx * self.ny


def make_grid(layout, vent=None, refine=1.0):
    """Build the mesh for a layout. refine=2 halves every cell size.

    Baseline resolution: 0.5 mm in x, 0.25 mm in y, >= 2 cells per layer.
    """
    xs = [0.0, dev.LENGTH, dev.DIE_X_CENTER]
    ys = [0.0, dev.HEIGHT]
    for r in layout["rects"]:
        xs += [r.x0, r.x1]
        ys += [r.y0, r.y1]
    if vent is not None:
        xs += list(vent)
    min_cells = max(1, int(round(2 * refine)))
    return Grid(_edges(xs, 0.5 * dev.MM / refine, min_cells),
                _edges(ys, 0.25 * dev.MM / refine, min_cells))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class Solution:
    """Everything worth knowing about one steady (or transient) state."""
    model: "Model2D"
    T: np.ndarray                 # cell temperatures, shape (ny, nx), C
    h: dict                       # converged heat transfer coefficients
    surf: dict                    # outer-surface temperatures per face, C
    q_out: dict                   # heat leaving through each boundary, W
    iterations: int = 0
    metrics: dict = field(default_factory=dict)


class Model2D:
    """Discretised cross-section for one design Variant."""

    def __init__(self, variant: dev.Variant, refine: float = 1.0):
        self.variant = variant
        self.layout = dev.build_layout(variant)
        self.vent = dev.vent_region(variant)
        self.grid = g = make_grid(self.layout, self.vent, refine)
        self.t_amb = variant.t_ambient
        D = dev.DEPTH

        # --- paint materials onto cells (later rectangles overwrite earlier)
        X, Y = np.meshgrid(g.xc, g.yc)
        self.kx = np.zeros((g.ny, g.nx))
        self.ky = np.zeros((g.ny, g.nx))
        self.rho_cp = np.zeros((g.ny, g.nx))
        self.qgen = np.zeros((g.ny, g.nx))           # W per cell
        self.region = np.zeros((g.ny, g.nx), dtype=int)
        self.region_names = []
        cell_area = np.outer(g.dy, g.dx)
        for idx, r in enumerate(self.layout["rects"]):
            m = (X > r.x0) & (X < r.x1) & (Y > r.y0) & (Y < r.y1)
            self.kx[m] = r.material.k_x
            self.ky[m] = r.material.k_y
            self.rho_cp[m] = r.material.rho_cp
            self.region[m] = idx
            self.region_names.append(r.name)
            if r.heat:
                self.qgen[m] = r.heat * cell_area[m] / (r.width * r.thick)

        # --- interior conductances (W/K), harmonic mean of half-cells
        dxh = g.dx[None, :] / (2 * self.kx)
        dyh = g.dy[:, None] / (2 * self.ky)
        self.Gx = (g.dy[:, None] * D) / (dxh[:, :-1] + dxh[:, 1:])   # (ny, nx-1)
        self.Gy = (g.dx[None, :] * D) / (dyh[:-1, :] + dyh[1:, :])   # (ny-1, nx)

        # --- half-cell resistances (per unit area) at each outer face
        self.r_half = dict(top=g.dy[-1] / (2 * self.ky[-1, :]),
                           bottom=g.dy[0] / (2 * self.ky[0, :]),
                           left=g.dx[0] / (2 * self.kx[:, 0]),
                           right=g.dx[-1] / (2 * self.kx[:, -1]))
        self.face_area = dict(top=g.dx * D, bottom=g.dx * D, left=g.dy * D, right=g.dy * D)
        # --- front/back walls (z = +/- DEPTH/2) are not in the 2D slice. We add
        # them back as straight fins hanging off the top and bottom walls: each
        # is a PC plate of thickness T_WALL, half the device height long,
        # cooled on its outer face. Fin theory [Incropera eq. 3.81, adiabatic
        # tip at mid-height] gives, per metre of device length and for the
        # front + back pair:
        #     G' = 2 * sqrt(h * k * t) * tanh(m * Lf),  m = sqrt(h / (k * t))
        # That conductance is shared by the wall cells in proportion to dx*dy.
        wall_rows = (g.yc > dev.HEIGHT - dev.T_WALL) | (g.yc < dev.T_WALL)
        I = np.arange(g.n).reshape(g.ny, g.nx)
        self.fin_cells = I[wall_rows, :].ravel()
        self.fin_weight = (np.outer(g.dy[wall_rows], g.dx) / dev.T_WALL).ravel()
        self.vent_mask = np.zeros(g.nx, dtype=bool)
        if self.vent is not None:
            self.vent_mask = (g.xc > self.vent[0]) & (g.xc < self.vent[1])

        # --- the fixed part of the conduction matrix
        self._K = self._conduction_matrix()

    # .....................................................................
    def idx(self, j, i):
        """Flattened index of cell (row j, column i)."""
        return j * self.grid.nx + i

    def _conduction_matrix(self):
        """Sparse matrix of the interior conductances (no boundary terms)."""
        g = self.grid
        I = np.arange(g.n).reshape(g.ny, g.nx)
        rows, cols, vals = [], [], []
        for a, b, G in ((I[:, :-1], I[:, 1:], self.Gx), (I[:-1, :], I[1:, :], self.Gy)):
            a, b, G = a.ravel(), b.ravel(), G.ravel()
            rows += [a, b, a, b]
            cols += [a, b, b, a]
            vals += [G, G, -G, -G]
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(g.n, g.n))

    def face_cells(self, face):
        """Flattened indices of the cells touching an outer face (or the fins)."""
        if face == "fins":
            return self.fin_cells
        g = self.grid
        I = np.arange(g.n).reshape(g.ny, g.nx)
        return {"top": I[-1, :], "bottom": I[0, :], "left": I[:, 0], "right": I[:, -1]}[face]

    # .....................................................................
    def h_coefficients(self, ts):
        """Film coefficients for every outer face cell, given surface temps.

        Convection uses the face-average temperature (it is a whole-plate
        buoyancy phenomenon); radiation uses the local surface temperature.
        The bottom face touches skin: a fixed effective conductance H_SKIN.
        """
        ta = self.t_amb
        g = self.grid
        top_mean = np.average(ts["top"], weights=g.dx)
        h_top_c = h_horizontal_up(top_mean, ta, dev.LENGTH * dev.DEPTH,
                                  2 * (dev.LENGTH + dev.DEPTH))
        h_top_c = np.where(self.vent_mask, dev.VENT_FACTOR * h_top_c, h_top_c)
        h = {"top": h_top_c + h_radiation(ts["top"], ta),
             "bottom": np.full(g.nx, dev.H_SKIN)}
        for face in ("left", "right"):
            mean = np.average(ts[face], weights=g.dy)
            h[face] = h_vertical(mean, ta, dev.HEIGHT) + h_radiation(ts[face], ta)
        h["top_conv"] = h_top_c
        # Fins (front/back walls) are vertical plates like the end walls.
        h["fin"] = 0.5 * (np.average(h["left"], weights=g.dy) + np.average(h["right"], weights=g.dy))
        return h

    @staticmethod
    def fin_conductance_per_length(h_fin):
        """Front + back wall fin conductance per metre of device length, W/m/K."""
        k, t, lf = dev.PC.k_x, dev.T_WALL, dev.HEIGHT / 2
        m = np.sqrt(h_fin / (k * t))
        return 2.0 * np.sqrt(h_fin * k * t) * np.tanh(m * lf)

    def boundary_conductance(self, h):
        """Cell-to-environment conductances, W/K.

        Outer faces: G = A / (r_half + 1/h). Fins: share of the fin conductance.
        """
        G = {f: self.face_area[f] / (self.r_half[f] + 1.0 / h[f])
             for f in ("top", "bottom", "left", "right")}
        G["fins"] = self.fin_conductance_per_length(h["fin"]) * self.fin_weight
        return G

    def sink_temp(self, face):
        """Environment temperature seen by a face: body core or room air."""
        return dev.T_BODY if face == "bottom" else self.t_amb

    def system(self, h):
        """Return (A, b) for steady conduction with the given film coefficients."""
        g = self.grid
        Gb = self.boundary_conductance(h)
        diag = np.zeros(g.n)
        b = self.qgen.ravel().copy()
        for f, G in Gb.items():
            cells = self.face_cells(f)
            np.add.at(diag, cells, G)
            np.add.at(b, cells, G * self.sink_temp(f))
        return self._K + sp.diags(diag), b

    def surface_temps(self, T, h):
        """Outer-surface temperature of every boundary face, C.

        The surface sits between the cell centre and the environment:
        T_s = T_inf + (T_c - T_inf) * (1/h) / (r_half + 1/h).
        """
        Tf = T.ravel()
        out = {}
        for f in ("top", "bottom", "left", "right"):
            tc = Tf[self.face_cells(f)]
            tinf = self.sink_temp(f)
            rc = 1.0 / h[f]
            out[f] = tinf + (tc - tinf) * rc / (self.r_half[f] + rc)
        return out

    def heat_out(self, T, h):
        """Heat leaving through each outer face (W); positive = out of device."""
        Tf = T.ravel()
        Gb = self.boundary_conductance(h)
        return {f: float(np.sum(G * (Tf[self.face_cells(f)] - self.sink_temp(f))))
                for f, G in Gb.items()}

    # .....................................................................
    def solve_steady(self, tol=1e-4, max_iter=60) -> Solution:
        """Steady state with Picard iteration on the temperature-dependent h."""
        g = self.grid
        guess = self.t_amb + 15.0
        ts = {"top": np.full(g.nx, guess), "bottom": np.full(g.nx, guess),
              "left": np.full(g.ny, guess), "right": np.full(g.ny, guess)}
        for it in range(1, max_iter + 1):
            h = self.h_coefficients(ts)
            A, b = self.system(h)
            T = spla.spsolve(A.tocsc(), b).reshape(g.ny, g.nx)
            ts_new = self.surface_temps(T, h)
            change = max(np.max(np.abs(ts_new[f] - ts[f])) for f in ts)
            ts = ts_new
            if change < tol:
                break
        else:
            raise RuntimeError("h iteration did not converge")
        sol = Solution(self, T, h, ts, self.heat_out(T, h), iterations=it)
        sol.metrics = metrics(sol)
        return sol

    # .....................................................................
    def mask(self, name):
        """Boolean (ny, nx) mask of the cells belonging to a named rectangle.

        Because rectangles are painted in order, this returns only the cells
        where that rectangle is actually visible.
        """
        idxs = [i for i, n in enumerate(self.region_names) if n == name]
        return np.isin(self.region, idxs)

    def volume_mean(self, T, mask):
        """Volume-weighted mean temperature over a mask, C."""
        w = np.outer(self.grid.dy, self.grid.dx)[mask]
        return float(np.sum(T[mask] * w) / np.sum(w))


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def heat_flux(sol: Solution):
    """Cell-centred heat-flux vector (qx, qy) in W/m^2.

    Fluxes are computed on cell faces from the same conductances the solver
    used (so they are exactly conservative), then averaged to cell centres.
    """
    m, g, T = sol.model, sol.model.grid, sol.T
    D = dev.DEPTH
    Gb = m.boundary_conductance(sol.h)
    fx = np.zeros((g.ny, g.nx + 1))   # W through each vertical face, +x direction
    fy = np.zeros((g.ny + 1, g.nx))   # W through each horizontal face, +y direction
    fx[:, 1:-1] = m.Gx * (T[:, :-1] - T[:, 1:])
    fy[1:-1, :] = m.Gy * (T[:-1, :] - T[1:, :])
    fx[:, 0] = -Gb["left"] * (T[:, 0] - m.t_amb)
    fx[:, -1] = Gb["right"] * (T[:, -1] - m.t_amb)
    fy[0, :] = -Gb["bottom"] * (T[0, :] - dev.T_BODY)
    fy[-1, :] = Gb["top"] * (T[-1, :] - m.t_amb)
    qx = 0.5 * (fx[:, :-1] + fx[:, 1:]) / (g.dy[:, None] * D)
    qy = 0.5 * (fy[:-1, :] + fy[1:, :]) / (g.dx[None, :] * D)
    return qx, qy


def bracket_field(sol: Solution):
    """Temperatures inside the bracket: (x centres, y centres, T[rows, cols])."""
    m = sol.model
    mask = m.mask("bracket")
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return m.grid.xc[cols], m.grid.yc[rows], sol.T[np.ix_(rows, cols)]


def metrics(sol: Solution) -> dict:
    """Headline numbers for one solution (all temperatures in C)."""
    m, T = sol.model, sol.T
    die = m.mask("SoC die")
    _, _, Tb = bracket_field(sol)
    s = sol.surf
    q_total = sum(sol.q_out.values())
    return {
        "T_junction": float(T[die].max()),
        "T_die_mean": m.volume_mean(T, die),
        "T_pcb_mean": m.volume_mean(T, m.mask("PCB")),
        "T_surface_peak": float(max(v.max() for v in s.values())),
        "T_top_peak": float(s["top"].max()),
        "T_skin_peak": float(s["bottom"].max()),
        "T_top_mean": float(np.average(s["top"], weights=m.grid.dx)),
        "T_skin_mean": float(np.average(s["bottom"], weights=m.grid.dx)),
        "bracket_dT": float(Tb.max() - Tb.min()),
        "bracket_mean": m.volume_mean(T, m.mask("bracket")),
        "Q_top": sol.q_out["top"],
        "Q_ends": sol.q_out["left"] + sol.q_out["right"],
        "Q_fins": sol.q_out["fins"],
        "Q_skin": sol.q_out["bottom"],
        "energy_error": (q_total - m.variant.power) / max(m.variant.power, 1e-12),
        "h_top_conv": float(np.min(sol.h["top_conv"])),
        "n_cells": m.grid.n,
    }


def solve(variant: dev.Variant, refine: float = 1.0) -> Solution:
    """Convenience: build the model for a variant and solve steady state."""
    return Model2D(variant, refine).solve_steady()
