"""
device.py -- the single source of truth for the wearable sensor module.

Everything you might want to tweak lives here: geometry, materials, heat
load, boundary conditions, and the design variants. Every other module reads
from this file and never hard-codes a physical number.

Coordinate system for the 2D cross-section (all lengths in metres, SI units):

    y ^   AMBIENT AIR (natural convection + radiation)
      |  +-------------------------------------------------------------+
      |  |######################  top wall (PC)  ######################|
      |  |#|       [spreader, variant-dependent, under the top wall]   |#|
      |  |#|                 [TIM]                                     |#|
      |  |#|                 [SoC die]                                 |#|
      |  |#|  [=================== PCB ===========================]   |#|
      |  |#|                 [pad][======= camera/IMU bracket =====]   |#|
      |  |#|                    internal air gap                       |#|
      |  |######################  bottom wall (PC)  ###################|
      |  +-------------------------------------------------------------+
      |   SKIN / BODY (contact boundary, comfort limit 43 C)
      +-------------------------------------------------------------------> x

The 2D model is a slice through the middle of the device, extruded over the
device depth DEPTH (the z direction, into the page). Parts that do not span
the full depth (the SoC die, its TIM, the bracket) are "depth-homogenised":
their layer gets an area-weighted mix of the part and the surrounding air, so
the through-thickness conductance and heat capacity of the real part are
preserved. See `homogenize()` and the README for the reasoning.

Reference abbreviations used in the citations below:
  [Incropera]  Incropera, DeWitt, Bergman, Lavine, "Fundamentals of Heat and
               Mass Transfer", 7th ed., Wiley 2011 (Tables A.1, A.3, A.4).
  [Lexan]      SABIC Lexan polycarbonate resin datasheets (typical values).
  [PGS]        Panasonic "PGS Graphite Sheet" product datasheet (100 um grade).
  [GapPad]     Typical silicone gap-pad / thermal-gel vendor datasheets
               (Bergquist/Henkel, Laird, Parker Chomerics class products).
  [Duck]       F. A. Duck, "Physical Properties of Tissue", Academic Press 1990.
  [IEC62368]   IEC 62368-1 touch-temperature guidance; 43 C is the widely used
               design limit for prolonged skin contact with a wearable.
"""

from dataclasses import dataclass, replace
from typing import Optional

MM = 1e-3  # metres per millimetre (all geometry below is written in mm * MM)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W/m^2/K^4 (CODATA 2018)
G_ACCEL = 9.81          # gravitational acceleration, m/s^2
KELVIN = 273.15         # 0 C in kelvin


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Material:
    """Thermal + mechanical properties of one material.

    k_x is the in-plane conductivity (along x in the cross-section) and k_y
    the through-thickness conductivity (along y). Isotropic materials simply
    have k_x == k_y. Graphite sheet and PCBs are strongly anisotropic.
    """
    name: str
    k_x: float    # in-plane thermal conductivity, W/m/K
    k_y: float    # through-plane thermal conductivity, W/m/K
    rho: float    # density, kg/m^3
    cp: float     # specific heat, J/kg/K
    cte: float    # linear coefficient of thermal expansion, 1/K
    note: str = ""

    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity rho*cp, J/m^3/K."""
        return self.rho * self.cp


def homogenize(part: Material, filler: Material, fraction: float, name: str) -> Material:
    """Mix a part with the surrounding filler over the device depth.

    The 2D model extrudes every layer across the full device depth. A 12 mm
    die inside an 80 mm deep device only occupies 15 % of that depth, so we
    replace its layer with a parallel (area-weighted) mix:

        k_eff     = f * k_part + (1 - f) * k_filler       (parallel paths)
        rho*cp_eff = f * rho*cp_part + (1 - f) * rho*cp_filler

    This keeps the real part's through-thickness conductance (k*A/t) and its
    heat capacity correct. What it cannot capture is 3D spreading in the
    depth direction -- a known limitation discussed in the README.
    """
    f = fraction
    rho_cp = f * part.rho_cp + (1 - f) * filler.rho_cp
    rho = f * part.rho + (1 - f) * filler.rho
    return Material(
        name=name,
        k_x=f * part.k_x + (1 - f) * filler.k_x,
        k_y=f * part.k_y + (1 - f) * filler.k_y,
        rho=rho,
        cp=rho_cp / rho,
        cte=part.cte,  # the structural part sets the expansion, not the air
        note=f"{part.name} homogenised over {f:.0%} of depth",
    )


def interface_effective_k(k_bulk: float, thickness: float, r_contact: float) -> float:
    """Fold the two contact resistances of a TIM into an effective conductivity.

    A TIM layer has bulk resistance t/k plus a contact resistance R''_c at each
    of its two faces (micro-roughness, incomplete wetting). In series:

        R''_total = t/k + 2*R''_c   ->   k_eff = t / R''_total

    r_contact is per interface, in m^2*K/W.
    """
    return thickness / (thickness / k_bulk + 2.0 * r_contact)


def pcb_effective_k(t_total, n_cu_layers, t_cu, coverage, k_cu=388.0, k_fr4=0.30):
    """Effective in-plane and through-plane conductivity of a multilayer PCB.

    Copper planes act as parallel paths in-plane (rule of mixtures):
        k_x = (n*t_cu*coverage*k_cu + (t_total - n*t_cu*coverage)*k_fr4) / t_total
    Through the thickness the dielectric layers are in series:
        k_y = t_total / ( t_dielectric/k_fr4 + n*t_cu/k_layer )
    with k_layer = coverage*k_cu + (1-coverage)*k_fr4 inside a copper layer.
    Thermal vias under the SoC would raise k_y locally; we ignore them
    (conservative). k_cu: annealed copper [Incropera]; k_fr4 ~0.3 W/m/K.
    Returns (k_x, k_y, copper volume fraction).
    """
    cu_area = n_cu_layers * t_cu * coverage
    k_x = (cu_area * k_cu + (t_total - cu_area) * k_fr4) / t_total
    k_layer = coverage * k_cu + (1 - coverage) * k_fr4
    t_diel = t_total - n_cu_layers * t_cu
    k_y = t_total / (t_diel / k_fr4 + n_cu_layers * t_cu / k_layer)
    return k_x, k_y, cu_area / t_total


# Air inside the sealed enclosure. Gaps are a few mm, so the Rayleigh number is
# far below the 1708 onset of cellular convection: the air conducts only. We add
# a radiation term between facing walls: h_rad = 4*sigma*T^3/(2/eps - 1)
# ~ 6 W/m^2/K at 320 K, eps = 0.9; across a ~3 mm gap that is k_rad ~ 0.018.
K_AIR = 0.0263  # W/m/K at 300 K [Incropera Table A.4]
AIR = Material("air (rad.-augmented)", K_AIR + 0.018, K_AIR + 0.018,
               1.16, 1007.0, 0.0, "conduction + gap radiation [Incropera A.4]")

PC = Material("polycarbonate", 0.20, 0.20, 1200.0, 1250.0, 65e-6,
              "enclosure plastic [Lexan]")

SILICON = Material("silicon", 130.0, 130.0, 2330.0, 712.0, 2.6e-6,
                   "148 W/m/K at 300 K, 119 at 400 K -> ~130 at die temp [Incropera A.1]")

COPPER = Material("copper C110", 388.0, 388.0, 8900.0, 385.0, 17e-6,
                  "pure Cu 401 W/m/K [Incropera A.1]; C110 sheet ~388")

PGS = Material("pyrolytic graphite sheet", 700.0, 15.0, 850.0, 850.0, 0.0,
               "100 um grade: 700 in-plane, ~15 through-plane, 0.85 g/cc [PGS]")

AL6061 = Material("aluminium 6061-T6", 167.0, 167.0, 2700.0, 896.0, 23.6e-6,
                  "[Incropera A.1] / ASM handbook values")

PAD_BRACKET = Material("bracket gap pad", 1.0, 1.0, 2500.0, 1000.0, 2e-4,
                       "compliant 1 W/m/K silicone pad [GapPad]")

PC_CONDUCTIVE = Material("thermally conductive PC", 2.0, 1.0, 1500.0, 1100.0, 30e-6,
                         "filled PC compound, ~2 W/m/K in-flow, ~1 through [vendor data]")

GAP_FILLER = Material("gap filler pad", 3.0, 3.0, 3000.0, 1000.0, 2e-4,
                      "soft 3 W/m/K silicone gap filler [GapPad]")

_pcb_kx, _pcb_ky, _pcb_cu = pcb_effective_k(t_total=1.2 * MM, n_cu_layers=8,
                                           t_cu=35e-6, coverage=0.5)
PCB = Material(
    "PCB (8L FR-4 + Cu)", _pcb_kx, _pcb_ky,
    rho=_pcb_cu * 8900 + (1 - _pcb_cu) * 1850,
    cp=(_pcb_cu * 8900 * 385 + (1 - _pcb_cu) * 1850 * 1100)
       / (_pcb_cu * 8900 + (1 - _pcb_cu) * 1850),
    cte=16e-6,
    note="8 x 1 oz copper at 50 % coverage, rule of mixtures",
)


@dataclass(frozen=True)
class TIMSpec:
    """A thermal interface material between the die and what sits above it."""
    name: str
    thickness: float  # bond-line thickness, m
    k_bulk: float     # datasheet bulk conductivity, W/m/K
    r_contact: float  # contact resistance per face, m^2*K/W
    rho: float
    cp: float

    def material(self) -> Material:
        """Return a Material whose k includes both contact resistances."""
        k = interface_effective_k(self.k_bulk, self.thickness, self.r_contact)
        return Material(self.name, k, k, self.rho, self.cp, 2e-4,
                        f"bulk {self.k_bulk} W/m/K + 2 x {self.r_contact:g} m2K/W contact")


# Generic gap pad: soft, absorbs a large tolerance stack, but it's thick and
# only modest k. Contact resistance ~0.2 cm^2K/W at low clamp pressure [GapPad].
TIM_GENERIC = TIMSpec("generic gap pad 1.5 W/mK, 0.5 mm",
                      0.5 * MM, 1.5, 2e-5, 2900.0, 1000.0)

# Dispensed thermal gel: thin bond line, high k, wets well. Needs tighter
# mechanical tolerances (it can't absorb a 0.4 mm gap variation) [GapPad].
TIM_GEL = TIMSpec("thermal gel 6 W/mK, 0.15 mm",
                  0.15 * MM, 6.0, 5e-6, 3200.0, 1000.0)


# ---------------------------------------------------------------------------
# Geometry, heat load and boundary conditions (baseline values)
# ---------------------------------------------------------------------------
LENGTH = 120 * MM        # device length along x (the cross-section width)
HEIGHT = 14 * MM         # device total thickness along y
DEPTH = 80 * MM          # out-of-plane depth (z); the 2D slice is extruded over it
T_WALL = 1.5 * MM        # enclosure wall thickness (top, bottom and ends)

POWER = 3.5              # SoC dissipation, W (mobile-class SoC under sustained load)
DIE_WIDTH = 12 * MM      # SoC footprint along x (square die/package, 12 x 12 mm)
DIE_DEPTH = 12 * MM      # SoC footprint along z (used for depth homogenisation)
DIE_THICK = 0.8 * MM     # silicon thickness
DIE_X_CENTER = 60 * MM   # SoC centred in the device

PCB_THICK = 1.2 * MM
PCB_X0, PCB_X1 = 4 * MM, 116 * MM          # PCB spans most of the interior

SPREADER_X0, SPREADER_X1 = 10 * MM, 110 * MM  # spreader footprint along x (100 mm)

BRACKET_LENGTH = 60 * MM                   # camera <-> IMU bracket
BRACKET_X0 = 55 * MM                       # IMU end, just under the SoC (hot end)
BRACKET_X1 = BRACKET_X0 + BRACKET_LENGTH   # camera end, near the device edge
BRACKET_THICK = 2.0 * MM
BRACKET_DEPTH = 20 * MM                    # bracket width along z
BRACKET_PAD_THICK = 0.3 * MM               # compliant pad bonding bracket to PCB
E_BRACKET = 68.9e9                         # Pa, Young's modulus of 6061-T6 (ASM handbook)
E_PCB = 20e9                               # Pa, in-plane modulus of FR-4 laminate (typ. 17-24 GPa)
T_CALIBRATION = 25.0                       # C, temperature at which the camera-IMU
                                           # extrinsics were calibrated in the factory

# Boundary conditions
T_AMBIENT = 25.0         # C, room air
EMISSIVITY = 0.90        # painted / textured plastic, IR emissivity [Incropera A.11]
T_BODY = 37.0            # C, body core temperature behind the tissue layer
# Skin-side boundary: the heat leaves through ~1 cm of perfused tissue
# (k ~ 0.37 W/m/K [Duck]) plus a strap/contact resistance. 1/(0.01/0.37) = 37,
# with contact ~0.002 m^2K/W in series -> ~35 W/m^2/K. With no device this BC
# gives a bare-skin temperature of ~34 C in 25 C air, which is physiological.
H_SKIN = 35.0            # W/m^2/K, effective conductance from device bottom to body core
SKIN_LIMIT = 43.0        # C, prolonged-contact comfort limit [IEC62368]
VENT_FACTOR = 2.5        # vented regions: convective h multiplied by this factor
VENT_HALF_WIDTH = 25 * MM  # vents cover +/- 25 mm of the top wall around the SoC


# ---------------------------------------------------------------------------
# Design variants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """One design option. Only the things that differ from baseline go here."""
    key: str
    label: str
    tim: TIMSpec = TIM_GENERIC
    spreader: Optional[Material] = None
    spreader_thick: float = 0.0
    vented: bool = False
    skin_spreader: Optional[Material] = None   # sheet on the inside of the bottom wall
    skin_spreader_thick: float = 0.0
    skin_bridge: bool = False                  # gap filler: bracket -> bottom wall
    top_shell: Material = None                 # top wall material (None = PC)
    power: float = POWER
    t_ambient: float = T_AMBIENT
    notes: str = ""


VARIANTS = [
    Variant("a", "a) Baseline", notes="plastic enclosure, no spreader, generic TIM"),
    Variant("b", "b) Cu spreader 0.5 mm", spreader=COPPER, spreader_thick=0.5 * MM),
    Variant("c", "c) Graphite (PGS) 0.1 mm", spreader=PGS, spreader_thick=0.1 * MM),
    Variant("d", "d) Thin high-k TIM", tim=TIM_GEL),
    Variant("e", "e) Vented enclosure", vented=True),
    # The design we end up recommending (see README "Recommendation"). Chosen
    # AFTER running a)-e) and the analysis studies, for these reasons:
    #  * the binding constraint is the outer-surface temperature, not the
    #    junction (Tj ~53 C vs a ~95 C throttle point);
    #  * two 100 um graphite sheets give ~98 % of the 0.5 mm copper sheet's
    #    sustainable power at ~4 % of its mass;
    #  * the generic gap pad is KEPT: a better TIM lowers Tj (non-binding) but
    #    sends more heat straight up into the hotspot over the SoC, which
    #    lowers the sustainable power. It also tolerates a looser stack-up.
    #  * the enclosure stays sealed (dust / sweat / camera window fogging).
    Variant("f", "f) 2x PGS, sealed (recommended)", spreader=PGS,
            spreader_thick=0.2 * MM,
            notes="recommended: two 100 um graphite sheets, generic TIM, sealed"),
]
RECOMMENDED_KEY = "f"


def variant_by_key(key: str) -> Variant:
    """Look up a variant by its single-letter key."""
    for v in VARIANTS:
        if v.key == key:
            return v
    raise KeyError(key)


def with_spreader(variant: Variant, material: Optional[Material], thickness: float) -> Variant:
    """Copy of a variant with a different spreader (used by the parametric sweep)."""
    if thickness <= 0:
        return replace(variant, spreader=None, spreader_thick=0.0)
    return replace(variant, spreader=material, spreader_thick=thickness)


# ---------------------------------------------------------------------------
# Layout: turn a Variant into a list of rectangles the solvers can mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rect:
    """An axis-aligned block of one material in the cross-section."""
    name: str
    material: Material
    x0: float
    x1: float
    y0: float
    y1: float
    heat: float = 0.0  # total heat generated in this block, W

    @property
    def width(self) -> float:
        """Extent along x, m."""
        return self.x1 - self.x0

    @property
    def thick(self) -> float:
        """Extent along y, m."""
        return self.y1 - self.y0


def build_layout(v: Variant):
    """Stack the layers from the top wall downwards and return the geometry.

    Stacking top-down keeps the die pressed against the TIM / spreader, so a
    thicker spreader or TIM simply pushes the die, PCB and bracket down into
    the air gap (as it would in the real product).

    Returns a dict with:
      rects      : list of Rect, painted in order (later ones overwrite)
      y_levels   : named y coordinates of interfaces (for plots / network)
      air_gap    : thickness of the remaining air gap below the bracket
    """
    top_inner = HEIGHT - T_WALL
    y_spr_bot = top_inner - v.spreader_thick
    y_tim_bot = y_spr_bot - v.tim.thickness
    y_die_bot = y_tim_bot - DIE_THICK
    y_pcb_bot = y_die_bot - PCB_THICK
    y_pad_bot = y_pcb_bot - BRACKET_PAD_THICK
    y_brk_bot = y_pad_bot - BRACKET_THICK
    y_skin_spr_top = T_WALL + v.skin_spreader_thick
    air_gap = y_brk_bot - y_skin_spr_top
    if air_gap < 0.5 * MM:
        raise ValueError(f"Stack-up does not fit: air gap {air_gap / MM:.2f} mm")

    die_frac = DIE_DEPTH / DEPTH
    brk_frac = BRACKET_DEPTH / DEPTH
    die_mat = homogenize(SILICON, AIR, die_frac, "SoC die (homog.)")
    tim_mat = homogenize(v.tim.material(), AIR, die_frac, "TIM (homog.)")
    brk_mat = homogenize(AL6061, AIR, brk_frac, "bracket (homog.)")
    pad_mat = homogenize(PAD_BRACKET, AIR, brk_frac, "bracket pad (homog.)")
    dx0 = DIE_X_CENTER - DIE_WIDTH / 2
    dx1 = DIE_X_CENTER + DIE_WIDTH / 2

    rects = [
        Rect("air", AIR, 0, LENGTH, 0, HEIGHT),
        Rect("bottom wall", PC, 0, LENGTH, 0, T_WALL),
        Rect("top wall", v.top_shell or PC, 0, LENGTH, top_inner, HEIGHT),
        Rect("left wall", PC, 0, T_WALL, 0, HEIGHT),
        Rect("right wall", PC, LENGTH - T_WALL, LENGTH, 0, HEIGHT),
        Rect("PCB", PCB, PCB_X0, PCB_X1, y_pcb_bot, y_die_bot),
        Rect("SoC die", die_mat, dx0, dx1, y_die_bot, y_tim_bot, heat=v.power),
        Rect("TIM", tim_mat, dx0, dx1, y_tim_bot, y_spr_bot),
        Rect("bracket pad", pad_mat, BRACKET_X0, BRACKET_X1, y_pad_bot, y_pcb_bot),
        Rect("bracket", brk_mat, BRACKET_X0, BRACKET_X1, y_brk_bot, y_pad_bot),
    ]
    if v.spreader is not None and v.spreader_thick > 0:
        rects.append(Rect("spreader", v.spreader, SPREADER_X0, SPREADER_X1,
                          y_spr_bot, top_inner))
    if v.skin_spreader is not None and v.skin_spreader_thick > 0:
        rects.append(Rect("skin spreader", v.skin_spreader, SPREADER_X0, SPREADER_X1,
                          T_WALL, y_skin_spr_top))
    if v.skin_bridge:
        bridge_mat = homogenize(GAP_FILLER, AIR, brk_frac, "skin bridge (homog.)")
        rects.append(Rect("skin bridge", bridge_mat, BRACKET_X0, BRACKET_X1,
                          y_skin_spr_top, y_brk_bot))

    y_levels = dict(top_inner=top_inner, spr_bot=y_spr_bot, tim_bot=y_tim_bot,
                    die_bot=y_die_bot, pcb_bot=y_pcb_bot, pad_bot=y_pad_bot,
                    brk_bot=y_brk_bot, skin_spr_top=y_skin_spr_top, bot_inner=T_WALL)
    return dict(rects=rects, y_levels=y_levels, air_gap=air_gap,
                die_x=(dx0, dx1), die_frac=die_frac, brk_frac=brk_frac)


def vent_region(v: Variant):
    """x-interval of the top wall that is vented, or None for a sealed device."""
    if not v.vented:
        return None
    return (DIE_X_CENTER - VENT_HALF_WIDTH, DIE_X_CENTER + VENT_HALF_WIDTH)


def added_mass_g(v: Variant) -> float:
    """Mass the variant adds over baseline (spreader + TIM change), grams.

    The spreader footprint is (SPREADER_X1-SPREADER_X0) x DEPTH; the TIM
    footprint is the die footprint.
    """
    m = 0.0
    if v.spreader is not None:
        m += v.spreader.rho * v.spreader_thick * (SPREADER_X1 - SPREADER_X0) * DEPTH
    if v.skin_spreader is not None:
        m += v.skin_spreader.rho * v.skin_spreader_thick * (SPREADER_X1 - SPREADER_X0) * DEPTH
    if v.skin_bridge:
        lay = build_layout(v)
        y = lay["y_levels"]
        m += GAP_FILLER.rho * (y["brk_bot"] - y["skin_spr_top"]) * BRACKET_LENGTH * BRACKET_DEPTH
    die_area = DIE_WIDTH * DIE_DEPTH
    m += (v.tim.rho * v.tim.thickness - TIM_GENERIC.rho * TIM_GENERIC.thickness) * die_area
    return m * 1e3
