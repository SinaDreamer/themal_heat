"""
plots.py -- every figure in results/.

Style rules used throughout (so the figures read as one set):
  * one colour per design variant, in a fixed order, used on every chart;
  * temperature fields use a single-hue (orange) light-to-dark ramp, never a
    rainbow, and paired plots share one colour scale;
  * limits are drawn as a labelled red line; text is always in neutral ink;
  * one y-axis per panel -- different units get different panels.
"""

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

import device as dev
from solver2d import heat_flux

# ---------------------------------------------------------------------------
# Palette and style
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LIMIT = "#d03b3b"  # status "critical": used only for the 43 C limit

# Categorical slots in fixed order: variant a..f -> slot 1..6
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
VARIANT_COLOR = {k: SERIES[i] for i, k in enumerate("abcdef")}

# Single-hue sequential ramp for temperature (orange family, light -> dark)
TEMP_CMAP = LinearSegmentedColormap.from_list(
    "temp_orange", ["#fdf3ea", "#f9d2b2", "#f3a877", "#e87d45", "#d45a26", "#a6401a", "#6b260d"])

MATERIAL_COLOR = {
    "air": "#f0efec", "polycarbonate": "#c3c2b7", "thermally conductive PC": "#a8a69c",
    "PCB": SERIES[5], "SoC die": SERIES[6], "TIM": SERIES[3], "spreader": SERIES[1],
    "bracket": SERIES[0], "bracket pad": SERIES[4], "skin spreader": SERIES[1],
    "skin bridge": SERIES[2],
}

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0, "savefig.dpi": 160,
})

V_EXAG = 2.0  # vertical exaggeration for full-device cross-section plots


def _save(fig, path):
    """Save a figure and close it."""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _limit_line(ax, y=dev.SKIN_LIMIT, label="43 °C skin-contact limit", horizontal=True):
    """Draw the comfort limit as a labelled red line."""
    if horizontal:
        ax.axhline(y, color=LIMIT, lw=1.2, ls=(0, (4, 2)), zorder=1)
        ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                    color=INK2, fontsize=8)


def _outline(ax, layout, yscale=1.0, lw=0.5, color=INK2, names=None):
    """Thin outlines of the solid parts (so contour plots show the stack-up)."""
    for r in layout["rects"]:
        if r.name == "air" or (names and r.name not in names):
            continue
        ax.add_patch(Rectangle((r.x0 / dev.MM, r.y0 / dev.MM * yscale), r.width / dev.MM,
                               r.thick / dev.MM * yscale, fill=False, lw=lw, ec=color,
                               alpha=0.55, zorder=4))


def _field_image(ax, sol, vmin, vmax, yscale=V_EXAG):
    """Filled temperature contours of a solution, in mm (y optionally stretched)."""
    g = sol.model.grid
    X, Y = np.meshgrid(g.xc / dev.MM, g.yc / dev.MM * yscale)
    levels = np.linspace(vmin, vmax, 31)
    cf = ax.contourf(X, Y, np.clip(sol.T, vmin, vmax), levels=levels, cmap=TEMP_CMAP)
    ax.set_xlim(0, dev.LENGTH / dev.MM)
    ax.set_ylim(0, dev.HEIGHT / dev.MM * yscale)
    ax.set_aspect("equal")
    ax.grid(False)
    yt = np.arange(0, dev.HEIGHT / dev.MM + 0.1, 2)
    ax.set_yticks(yt * yscale)
    ax.set_yticklabels([f"{t:.0f}" for t in yt])
    return cf


def _mat_key(r):
    """Colour key for a layout rectangle: walls by material, parts by name."""
    if r.name == "air":
        return "air"
    if r.name.endswith("wall"):
        return r.material.name
    return r.name


# ---------------------------------------------------------------------------
# 1. Cross-section schematic
# ---------------------------------------------------------------------------
def plot_schematic(path, variant):
    """Annotated stack-up of the device (vertical scale exaggerated) + zoom."""
    lay = dev.build_layout(variant)
    ys = 3.0
    fig = plt.figure(figsize=(12, 7.2))
    ax = fig.add_axes([0.05, 0.42, 0.9, 0.5])
    for r in lay["rects"]:
        color = MATERIAL_COLOR.get(_mat_key(r), MUTED)
        ax.add_patch(Rectangle((r.x0 / dev.MM, r.y0 / dev.MM * ys), r.width / dev.MM,
                               r.thick / dev.MM * ys, fc=color, ec=SURFACE if r.name != "air" else AXIS,
                               lw=0.6))
    ax.set_xlim(-2, dev.LENGTH / dev.MM + 2)
    ax.set_ylim(-14, dev.HEIGHT / dev.MM * ys + 14)
    ax.set_aspect("equal")
    ax.grid(False)
    yt = np.arange(0, dev.HEIGHT / dev.MM + 0.1, 2)
    ax.set_yticks(yt * ys)
    ax.set_yticklabels([f"{t:.0f}" for t in yt])
    ax.set_xlabel("x (mm)")
    ax.set_ylabel(f"y (mm, ×{ys:.0f} vertical)")
    ax.set_title(f"1 · Device cross-section — {variant.label}   "
                 f"({dev.LENGTH / dev.MM:.0f} × {dev.DEPTH / dev.MM:.0f} × {dev.HEIGHT / dev.MM:.0f} mm, "
                 f"{variant.power:.1f} W SoC)")

    def note(text, xy_mm, xytext_mm):
        ax.annotate(text, xy=(xy_mm[0], xy_mm[1] * ys), xytext=(xytext_mm[0], xytext_mm[1]),
                    fontsize=8.5, color=INK, ha="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))

    y = {k: v / dev.MM for k, v in lay["y_levels"].items()}
    top = dev.HEIGHT / dev.MM
    note("Top wall: PC 1.5 mm, k = 0.20\nnatural convection + radiation to 25 °C air",
         (20, top - 0.4), (22, top * ys + 9))
    if variant.spreader is not None:
        note(f"Spreader: {variant.spreader.name}\n{variant.spreader_thick / dev.MM:.2f} mm, 100 mm wide",
             (95, y["spr_bot"] + 0.05), (98, top * ys + 9))
    note(f"SoC die 12 mm, {variant.power:.1f} W", (dev.DIE_X_CENTER / dev.MM, (y["tim_bot"] + y["die_bot"]) / 2),
         (58, top * ys + 9))
    note(f"PCB 1.2 mm  (k∥ = {dev.PCB.k_x:.0f}, k⊥ = {dev.PCB.k_y:.2f} W/m·K)",
         (20, (y["die_bot"] + y["pcb_bot"]) / 2), (24, 20))
    note("Camera–IMU bracket, Al 6061, 60 mm\n(IMU end under SoC, camera end at edge)",
         (100, (y["pad_bot"] + y["brk_bot"]) / 2), (96, -11))
    note("Internal air gap (conduction + radiation)", (60, (y["brk_bot"] + y["bot_inner"]) / 2), (40, 9))
    note("Bottom wall: PC 1.5 mm  →  skin contact\n(h = 35 W/m²K to 37 °C core, limit 43 °C)",
         (20, 0.5), (24, -12))

    # zoom inset at true scale around the die stack
    axz = fig.add_axes([0.05, 0.05, 0.42, 0.3])
    for r in lay["rects"]:
        axz.add_patch(Rectangle((r.x0 / dev.MM, r.y0 / dev.MM), r.width / dev.MM, r.thick / dev.MM,
                                fc=MATERIAL_COLOR.get(_mat_key(r), MUTED), ec=SURFACE, lw=0.6))
    axz.set_xlim(44, 76)
    axz.set_ylim(y["brk_bot"] - 0.5, top)
    axz.set_aspect("equal")
    axz.grid(False)
    axz.set_title("Zoom on the SoC stack (true scale)", fontsize=9.5)
    axz.set_xlabel("x (mm)")
    axz.set_ylabel("y (mm)")

    # legend / property table
    axl = fig.add_axes([0.52, 0.03, 0.45, 0.32])
    axl.axis("off")
    rows = [("polycarbonate", "PC enclosure", dev.PC), ("PCB", "PCB 8L FR-4/Cu", dev.PCB),
            ("SoC die", "silicon die", dev.SILICON), ("TIM", "TIM (eff.)", variant.tim.material()),
            ("bracket", "Al 6061-T6", dev.AL6061), ("bracket pad", "bracket pad", dev.PAD_BRACKET)]
    if variant.spreader is not None:
        rows.insert(3, ("spreader", "PGS graphite" if variant.spreader is dev.PGS else "copper",
                        variant.spreader))
    axl.text(0, 1.0, "Material         k_x / k_y (W/mK)   ρ (kg/m³)   cp (J/kg·K)   CTE (ppm/K)",
             fontsize=8.2, color=INK2, family="monospace", va="top")
    for i, (key, label, m) in enumerate(rows):
        yy = 0.88 - i * 0.12
        axl.add_patch(Rectangle((0, yy - 0.07), 0.025, 0.08, fc=MATERIAL_COLOR[key], ec=AXIS,
                                transform=axl.transAxes))
        axl.text(0.04, yy, f"{label:<16} {m.k_x:>6.2f} / {m.k_y:<6.2f}  {m.rho:>7.0f}  "
                           f"{m.cp:>8.0f}   {m.cte * 1e6:>8.1f}",
                 fontsize=8.2, color=INK, family="monospace", va="top")
    axl.text(0, 0.88 - len(rows) * 0.12 - 0.02,
             "Die, TIM and bracket layers are depth-homogenised with air in the solver\n"
             "(die 12/80 mm, bracket 20/80 mm of the depth). TIM k includes contact resistance.",
             fontsize=7.8, color=INK2, va="top")
    _save(fig, path)


# ---------------------------------------------------------------------------
# 2. Resistance network
# ---------------------------------------------------------------------------
def _resistor(ax, p0, p1, label, value, flow, side="right"):
    """Draw a zig-zag resistor between two points with a text label."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    length = np.linalg.norm(d)
    u = d / length
    n = np.array([-u[1], u[0]])
    a, b = 0.32, 0.68
    zig = [p0, p0 + a * d]
    k = 8
    for i in range(1, k):
        t = a + (b - a) * i / k
        zig.append(p0 + t * d + n * 0.012 * (1 if i % 2 else -1))
    zig += [p0 + b * d, p1]
    zig = np.array(zig)
    ax.plot(zig[:, 0], zig[:, 1], color=INK2, lw=1.3, zorder=2)
    mid = p0 + 0.5 * d
    text = f"{label}\n{value:.2f} K/W  ·  {flow:.2f} W" if label else f"{value:.2f} K/W  ·  {flow:.2f} W"
    if abs(u[0]) > abs(u[1]):   # horizontal resistor: label above (or below) it
        below = side == "below"
        ax.text(mid[0], mid[1] + (-0.03 if below else 0.03), text, ha="center",
                va="top" if below else "bottom", fontsize=8.3, color=INK)
    else:
        off = 0.03 if side == "right" else -0.03
        ax.text(mid[0] + off, mid[1], text, ha="left" if side == "right" else "right",
                va="center", fontsize=8.3, color=INK)


def _node(ax, xy, name, temp, fixed=False):
    """Draw a network node with its temperature."""
    ax.add_patch(Circle(xy, 0.022, fc=INK if fixed else SURFACE, ec=INK, lw=1.4, zorder=3))
    ax.text(xy[0] - 0.035, xy[1], f"{name}\n{temp:.1f} °C", ha="right", va="center",
            fontsize=8.8, color=INK, fontweight="bold")


def plot_network(path, net, variant, sol2d=None):
    """Labelled resistance network with computed values and node temperatures."""
    fig, ax = plt.subplots(figsize=(10.5, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    T = net.T
    R = {r.name: r for r in net.resistors}
    F = net.flows
    pos = {"Amb": (0.42, 0.98), "To": (0.42, 0.80), "S": (0.42, 0.63), "J": (0.42, 0.47),
           "P": (0.42, 0.33), "Bi": (0.42, 0.20), "Bo": (0.42, 0.08), "Body": (0.42, -0.03),
           "AmbB": (0.86, 0.08), "AmbT": (0.86, 0.80)}
    names = {"Amb": "Ambient air", "To": "Top surface", "S": "Top plate\n(inner face)",
             "J": "SoC junction", "P": "PCB", "Bi": "Bottom wall\n(inner)", "Bo": "Skin-side\nsurface",
             "Body": "Body core"}

    _resistor(ax, pos["To"], pos["Amb"], "R_conv,top", R["R_conv,top"].R, F["R_conv,top"], side="left")
    _resistor(ax, pos["S"], pos["To"], "R_shell", R["R_shell"].R, F["R_shell"])
    _resistor(ax, pos["J"], pos["S"], "R_die+TIM+spread", R["R_die+TIM+spread"].R, F["R_die+TIM+spread"])
    ax.text(0.47, 0.515, R["R_die+TIM+spread"].detail, fontsize=7.6, color=INK2)
    _resistor(ax, pos["J"], pos["P"], "R_die+PCB", R["R_die+PCB"].R, F["R_die+PCB"])
    ax.text(0.47, 0.375, R["R_die+PCB"].detail, fontsize=7.6, color=INK2)
    _resistor(ax, pos["P"], pos["Bi"], "R_gap,bottom", R["R_gap,bottom"].R, F["R_gap,bottom"])
    _resistor(ax, pos["Bi"], pos["Bo"], "R_wall,bottom", R["R_wall,bottom"].R, F["R_wall,bottom"])
    _resistor(ax, pos["Bo"], pos["Body"], "R_skin", R["R_skin"].R, F["R_skin"])
    # cross link P -> S drawn on the right
    xr = 0.80
    ax.plot([pos["P"][0], xr], [pos["P"][1]] * 2, color=INK2, lw=1.3)
    ax.plot([pos["S"][0], xr], [pos["S"][1]] * 2, color=INK2, lw=1.3)
    _resistor(ax, (xr, pos["P"][1]), (xr, pos["S"][1]), "R_air,top", R["R_air,top"].R, F["R_air,top"])
    # fins to ambient on the left
    _resistor(ax, (pos["To"][0] + 0.022, pos["To"][1]), pos["AmbT"], "R_fins,top (end + front/back walls)",
              R["R_fins,top"].R, F["R_fins,top"])
    _resistor(ax, (pos["Bo"][0] + 0.022, pos["Bo"][1]), pos["AmbB"], "R_fins,bottom",
              R["R_fins,bottom"].R, F["R_fins,bottom"], side="below")

    for k in ("To", "S", "J", "P", "Bi", "Bo"):
        _node(ax, pos[k], names[k], T[k])
    _node(ax, pos["Amb"], names["Amb"], T["Amb"], fixed=True)
    _node(ax, pos["Body"], names["Body"], T["Body"], fixed=True)
    for k in ("AmbT", "AmbB"):
        ax.add_patch(Circle(pos[k], 0.018, fc=INK, ec=INK, zorder=3))
        ax.text(pos[k][0], pos[k][1] - 0.04, f"Ambient\n{T['Amb']:.0f} °C", ha="center", va="top",
                fontsize=8, color=INK2)
    ax.annotate(f"{variant.power:.1f} W", xy=pos["J"], xytext=(0.62, 0.47), fontsize=10,
                color=INK, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_title(f"2 · Thermal resistance network — {variant.label}\n"
                 "resistance (K/W) · heat flow (W); node temperatures from nodal analysis",
                 loc="left")
    if sol2d is not None:
        m = sol2d.metrics
        ax.text(0.98, 1.0,
                "Same nodes from the 2D solver:\n"
                f"junction (die mean) {m['T_die_mean']:.1f} °C\n"
                f"top surface (mean) {m['T_top_mean']:.1f} °C\n"
                f"PCB (mean) {m['T_pcb_mean']:.1f} °C\n"
                f"skin-side surface (mean) {m['T_skin_mean']:.1f} °C",
                fontsize=8.3, color=INK2, va="top", ha="right")
    _save(fig, path)


# ---------------------------------------------------------------------------
# 3. Contours baseline vs best (shared scale)
# ---------------------------------------------------------------------------
def plot_contours(path, sol_a, sol_b):
    """Temperature fields of two designs on one colour scale."""
    vmin = min(sol_a.T.min(), sol_b.T.min())
    vmax = max(sol_a.T.max(), sol_b.T.max())
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.4))
    for ax, sol in zip(axes, (sol_a, sol_b)):
        cf = _field_image(ax, sol, vmin, vmax)
        _outline(ax, sol.model.layout, yscale=V_EXAG)
        m = sol.metrics
        ax.set_title(f"{sol.model.variant.label}:  junction {m['T_junction']:.1f} °C · "
                     f"peak surface {m['T_surface_peak']:.1f} °C · skin side {m['T_skin_peak']:.1f} °C",
                     fontsize=9.5)
        ax.set_ylabel(f"y (mm, ×{V_EXAG:.0f})")
        g = sol.model.grid
        i = np.argmax(sol.surf["top"])
        ax.plot(g.xc[i] / dev.MM, dev.HEIGHT / dev.MM * V_EXAG + 0.9, marker="v", ms=7, color=INK,
                clip_on=False, zorder=6)
    axes[-1].set_xlabel("x (mm)")
    cb = fig.colorbar(cf, ax=axes, shrink=0.85, pad=0.015, ticks=np.arange(np.ceil(vmin), vmax, 2))
    cb.set_label("Temperature (°C)   ▼ = surface hotspot")
    cb.outline.set_edgecolor(AXIS)
    fig.suptitle("3 · Steady-state temperature, same colour scale", x=0.05, ha="left",
                 fontweight="bold", fontsize=11)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 4. Heat-flux vectors on the best variant
# ---------------------------------------------------------------------------
def plot_flux(path, sol):
    """Isotherms and heat-flux directions around the SoC, true scale."""
    g = sol.model.grid
    qx, qy = heat_flux(sol)
    x0, x1 = 20.0, 100.0
    fig, ax = plt.subplots(figsize=(12, 3.9))
    X, Y = np.meshgrid(g.xc / dev.MM, g.yc / dev.MM)
    vmin, vmax = sol.T.min(), sol.T.max()
    cf = ax.contourf(X, Y, sol.T, levels=np.linspace(vmin, vmax, 31), cmap=TEMP_CMAP)
    cs = ax.contour(X, Y, sol.T, levels=np.arange(np.ceil(vmin), vmax, 1.0), colors=INK2,
                    linewidths=0.45, alpha=0.7)
    ax.clabel(cs, cs.levels[::3], fontsize=6.5, fmt="%.0f °C")
    _outline(ax, sol.model.layout, lw=0.6, color=INK)
    # regular sample of the flux field; arrows show direction, length ~ log magnitude
    xs = np.linspace(x0 + 1, x1 - 1, 34) * dev.MM
    ysmp = np.linspace(0.6, dev.HEIGHT / dev.MM - 0.3, 10) * dev.MM
    ii = np.clip(np.searchsorted(g.xe, xs) - 1, 0, g.nx - 1)
    jj = np.clip(np.searchsorted(g.ye, ysmp) - 1, 0, g.ny - 1)
    J, I = np.meshgrid(jj, ii, indexing="ij")
    u, v = qx[J, I], qy[J, I]
    mag = np.hypot(u, v)
    scale = np.log10(np.maximum(mag, 1.0)) / np.log10(mag.max())
    ax.quiver(g.xc[I] / dev.MM, g.yc[J] / dev.MM, u / mag * scale, v / mag * scale,
              angles="uv", scale=40, width=0.0018, color=INK, zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, dev.HEIGHT / dev.MM)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm, true scale)")
    cb = fig.colorbar(cf, cax=ax.inset_axes([1.01, 0, 0.012, 1]), ticks=np.arange(np.ceil(vmin), vmax, 2))
    cb.set_label("Temperature (°C)")
    cb.outline.set_edgecolor(AXIS)
    m = sol.metrics
    ax.set_title(f"4 · Heat-flux paths — {sol.model.variant.label}   "
                 f"(arrows: direction, length ∝ log|q|; lines: 1 °C isotherms)\n"
                 f"Heat out: top {m['Q_top']:.2f} W · end walls {m['Q_ends']:.2f} W · "
                 f"front/back walls {m['Q_fins']:.2f} W · into the body {m['Q_skin']:.2f} W",
                 fontsize=9.5)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 5. Variant comparison
# ---------------------------------------------------------------------------
def plot_variants(path, results):
    """Four small multiples: junction, peak surface, bracket dT, sustainable power."""
    keys = list(results)
    short = {"a": "Baseline", "b": "Cu 0.5 mm", "c": "PGS 0.1 mm", "d": "Thin TIM",
             "e": "Vented", "f": "2×PGS sealed\n(recommended)"}
    labels = [f"{k})\n{short.get(k, k)}" for k in keys]
    colors = [VARIANT_COLOR[k] for k in keys]
    panels = [
        ("Junction temperature (°C)", [results[k]["m"]["T_junction"] for k in keys], "{:.1f}", None),
        ("Peak outer-surface temperature (°C) — dashed: 43 °C limit",
         [results[k]["m"]["T_surface_peak"] for k in keys],
         "{:.1f}", dev.SKIN_LIMIT),
        ("Bracket temperature spread ΔT (K)", [results[k]["exp"]["dT_max"] for k in keys], "{:.2f}", None),
        ("Sustainable power at 43 °C surface (W) — dashed: 3.5 W target",
         [results[k]["p_max"] for k in keys], "{:.2f}",
         dev.POWER),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, vals, fmt, ref) in zip(axes.ravel(), panels):
        x = np.arange(len(keys))
        bars = ax.bar(x, vals, color=colors, width=0.66, edgecolor=SURFACE, linewidth=2, zorder=2)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val, fmt.format(val), ha="center", va="bottom",
                    fontsize=8, color=INK, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        lo = 0 if ref == dev.POWER or "ΔT" in title else 30
        ax.set_ylim(lo, max(vals + [ref or 0]) * 1.1 if lo == 0 else max(vals) + 4)
        if ref is not None:
            ax.axhline(ref, color=LIMIT, lw=1.2, ls=(0, (4, 2)), zorder=3)
    fig.suptitle("5 · Design variants at 3.5 W, 25 °C room (sustainable power: steady state, any surface ≤ 43 °C)",
                 x=0.05, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


# ---------------------------------------------------------------------------
# 6. Spreader thickness sweep
# ---------------------------------------------------------------------------
def plot_sweep(path, sweep):
    """Junction temperature, surface temperature and added mass vs thickness."""
    color = {dev.COPPER.name: VARIANT_COLOR["b"], dev.PGS.name: VARIANT_COLOR["c"]}
    short = {dev.COPPER.name: "Copper", dev.PGS.name: "Graphite (PGS)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for name, arr in sweep.items():
        t, tj, ts, mass = arr.T
        for ax, y in zip(axes, (tj, ts, mass)):
            ax.plot(t, y, color=color[name], marker="o", ms=4.5, label=short[name])
    _limit_line(axes[1])
    axes[0].set_title("Junction temperature (°C)")
    axes[1].set_title("Peak outer-surface temperature (°C)")
    axes[2].set_title("Added mass (g)")
    for ax in axes:
        ax.set_xlabel("Spreader thickness (mm)")
    axes[0].legend(loc="upper right")
    axes[1].set_ylim(top=max(arr[:, 2].max() for arr in sweep.values()) + 1)
    fig.suptitle("6 · Spreader thickness sweep (generic TIM, sealed, 3.5 W) — the benefit saturates; "
                 "in-plane conductance k·t is what matters",
                 x=0.05, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


# ---------------------------------------------------------------------------
# 7. Transient warm-up
# ---------------------------------------------------------------------------
def plot_transient(path, runs, burst):
    """Junction and peak-surface temperature vs time for two designs."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, field, title in ((axes[0], "T_junction", "SoC junction temperature"),
                             (axes[1], "T_surface", "Peak outer-surface temperature")):
        for key, r in runs.items():
            label = dev.variant_by_key(key).label
            ax.plot(r.t / 60, getattr(r, field), color=VARIANT_COLOR[key], label=label)
            ax.axhline(getattr(r, field)[-1], color=VARIANT_COLOR[key], lw=0.8, ls=":", zorder=1)
        ax.set_title(f"{title} (°C)")
        ax.set_xlabel("Time after power-on (min)")
        ax.set_xlim(0, 60)
    _limit_line(axes[1])
    for key, tb in burst.items():
        if np.isfinite(tb):
            axes[1].plot(tb / 60, dev.SKIN_LIMIT, "o", ms=7, color=VARIANT_COLOR[key],
                         mec=SURFACE, mew=2, zorder=5)
            axes[1].annotate(f"{tb / 60:.1f} min", xy=(tb / 60, dev.SKIN_LIMIT), xytext=(6, -14),
                             textcoords="offset points", fontsize=8.3, color=INK)
    axes[0].legend(loc="lower right")
    axes[1].text(0.98, 0.04, "Kink in the first minutes: early on the hottest surface is the\n"
                 "skin side (warmed by the 37 °C body); then the top surface over the SoC takes over.",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7.8, color=INK2)
    fig.suptitle("7 · Warm-up at 3.5 W from a cold start (dotted: steady-state value). "
                 "Dots: time until the surface reaches 43 °C = burst budget",
                 x=0.05, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


# ---------------------------------------------------------------------------
# 8. Bracket expansion
# ---------------------------------------------------------------------------
def plot_bracket(path, results, keys):
    """Temperature, axial growth and thermal bow along the camera-IMU bracket."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for key in keys:
        e = results[key]["exp"]
        x = e["x"] / dev.MM
        c = VARIANT_COLOR[key]
        lab = results[key]["variant"].label
        axes[0].plot(x, e["T_mean"], color=c, label=lab)
        axes[1].plot(x, e["u"] * 1e6, color=c, label=lab)
        axes[2].plot(x, e["w"] * 1e9, color=c, label=lab)
        axes[1].annotate(f"{e['dL_um']:.1f} µm", xy=(x[-1], e["u"][-1] * 1e6), xytext=(-4, 6),
                         textcoords="offset points", ha="right", fontsize=8.3, color=INK)
    axes[0].set_title("Bracket temperature (°C)")
    axes[1].set_title(f"Axial growth from IMU end (µm), vs {dev.T_CALIBRATION:.0f} °C calibration")
    axes[2].set_title("Thermal bow, free bracket (nm)")
    axes[2].set_ylim(top=max(axes[2].get_ylim()[1], 12))
    for ax in axes:
        ax.set_xlabel("Distance from IMU end (mm)")
    axes[0].legend(loc="upper right", fontsize=8)
    eb, ef = results[keys[0]]["exp"], results[keys[-1]]["exp"]
    axes[2].text(0.5, 0.97,
                 "Camera-to-IMU rotation\n"
                 f"free (kinematic) mount: {eb['rotation_urad']:.1f} → {ef['rotation_urad']:.1f} µrad "
                 f"({eb['rotation_arcsec']:.2f} → {ef['rotation_arcsec']:.2f}″)\n"
                 f"if bonded rigidly to FR-4: ≈ {ef['bonded_rotation_urad'] / 1e3:.1f} mrad "
                 f"({ef['bonded_rotation_arcsec']:.0f}″)",
                 transform=axes[2].transAxes, fontsize=8, color=INK, va="top", ha="center")
    fig.suptitle("8 · Camera–IMU bracket: axial growth dominates; gradient-driven bow is tiny "
                 "when the mount lets the bracket expand freely",
                 x=0.05, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path)


# ---------------------------------------------------------------------------
# 9. Grid convergence
# ---------------------------------------------------------------------------
def plot_convergence(path, rows, extrap):
    """Junction and surface temperature vs number of cells."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, title, ex in ((axes[0], 2, "Junction temperature (°C)", extrap[0]),
                               (axes[1], 3, "Peak surface temperature (°C)", extrap[1])):
        ax.plot(rows[:, 1], rows[:, col], marker="o", color=SERIES[0])
        ax.axhline(ex, color=MUTED, lw=1, ls=":")
        ax.annotate("Richardson extrapolation", xy=(rows[0, 1], ex), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
        ax.annotate("default mesh", xy=(rows[1, 1], rows[1, col]), xytext=(8, 8),
                    textcoords="offset points", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("Number of cells")
        ax.set_title(title)
    fig.suptitle("9 · Grid convergence, baseline design (each step halves the cell size)",
                 x=0.05, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path)
