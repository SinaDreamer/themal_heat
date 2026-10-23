# Thermal Path Design Lab: Wearable Sensor Module

A self-contained Python project that models, simulates and compares the thermal path of a
compact wearable data-capture module (SoC + cameras + IMU in a sealed plastic enclosure
worn against the body). It answers three questions:

1. **Where does the 3.5 W go, and how hot does the SoC get?** (1D resistance network, 2D solver, transient)
2. **Is the device comfortable to wear?** (peak surface temperature against the 43 °C limit)
3. **Does heat hurt sensor accuracy?** (camera–IMU bracket expansion and misalignment)

Everything is pure Python (numpy, scipy, matplotlib). There's no CFD and no network access, and one
command rebuilds every number and plot.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py          # ~20 s, writes everything into results/
```

`run_all.py` exits with status 1 if any sanity check fails. Results are deterministic: two
runs produce byte-identical files.

| File | What it is |
|---|---|
| `device.py` | **Every input lives here.** Geometry, materials (k, ρ, cp, CTE, with sources), heat load, boundary conditions, design variants |
| `convection.py` | Natural-convection and radiation coefficients (textbook correlations) |
| `network1d.py` | Thermal resistance network with exact strip spreading resistance (the hand-calc model) |
| `solver2d.py` | 2D finite-volume conduction solver for the cross-section |
| `transient.py` | Implicit time stepping for warm-up |
| `expansion.py` | Bracket thermal expansion, bow and camera-to-IMU rotation |
| `analysis.py` | Studies: variants, sustainable power, spreader sweep, grid convergence, gap closers |
| `sanity.py` | 47 automatic checks (energy balance, grid convergence, 1D vs 2D, physics) |
| `plots.py` | All figures |
| `results/` | PNGs, `summary.md` (all tables), `variants.csv`, `sanity_checks.txt` |

---

## 1. Thermal path theory in plain language

**Heat behaves like current, and temperature like voltage.** The SoC injects 3.5 W. That power must
reach the environment, either the room air at 25 °C or the body at 37 °C. Every material it
crosses has a *thermal resistance* R (K/W), and the temperature rise is ΔT = Q·R, which is Ohm's
law for heat. Thermal design is therefore about arranging resistances so the heat takes the
path you want and nothing on that path gets too hot.

The resistance types that matter here:

| Mechanism | Formula | Notes for this device |
|---|---|---|
| Conduction through a layer | R = t / (k·A) | 1.5 mm of polycarbonate (k = 0.2) is a surprisingly big resistance: about 2.5 K at full power |
| Contact / TIM | R = (t/k + 2R''c) / A | The two interfaces of a TIM matter as much as its bulk k when it's thin |
| Spreading | R_sp (Fourier series) | Extra resistance when heat enters a plate through a small footprint. **In the baseline this is 47 K/W**, so a plastic wall can't spread heat at all |
| Convection + radiation | R = 1 / (h·A) | Natural convection h ≈ 7 W/m²K plus radiation ≈ 6.5 W/m²K. Radiation is half the cooling, which is why emissivity matters |
| Fins | G = √(hkt)·w·tanh(mL) | The end walls and front/back walls are poor plastic fins |
| Skin | R = 1 / (h_skin·A) | Perfused tissue conducts heat to the 37 °C core (h ≈ 35 W/m²K) |

**Two different problems hide in "too hot":**

* **Peak vs. average.** The *average* outer-surface temperature is fixed by how much surface area
  and h you have: ΔT_avg ≈ Q / (h·A_total). No internal design changes this. The *peak* (the
  hotspot over the SoC) depends on how well heat spreads sideways before it reaches the surface.
  Spreaders fix peaks. Only area, h, or power fixes the average.
* **Junction vs. skin.** Lowering the junction temperature (a better TIM, for instance) is pointless if
  the junction is not the binding constraint. In this device it isn't: Tj ≈ 53 °C against a
  ~95 °C throttle point. The binding constraint is the 43 °C surface.

**Sensor accuracy.** Temperature changes geometry. A 60 mm aluminium bracket heated 25 K above its
calibration temperature grows α·L·ΔT ≈ 35 µm. A temperature difference *through* the bracket
makes it bow like a bimetal strip, rotating the camera relative to the IMU by
θ ≈ α·L·ΔT_through / h.

---

## 2. The device model (`device.py`)

**Envelope:** 120 × 80 × 14 mm sealed polycarbonate enclosure with 1.5 mm walls. Worn with the
bottom face on the body and the top face up in still room air.

**Stack-up from the top down** (all parameters are in `device.py`):

| Layer | Material | Thickness | k (W/m·K) | Source / reasoning |
|---|---|---|---|---|
| Top wall | Polycarbonate | 1.5 mm | 0.20 | SABIC Lexan datasheet |
| Spreader (variant) | Cu C110 / pyrolytic graphite | 0.1–1.0 mm | 388 / 700 in-plane, 15 through | Incropera A.1; Panasonic PGS 100 µm grade |
| TIM | Gap pad 1.5 W/mK, 0.5 mm (or gel 6 W/mK, 0.15 mm) | | 1.34 effective (incl. 2 × 2e-5 m²K/W contact) | Vendor datasheet class values |
| SoC die | Silicon, 12 × 12 mm, 0.8 mm, **3.5 W** | | 130 | Incropera A.1 (148 at 300 K, 119 at 400 K) |
| PCB | 8-layer FR-4, 1 oz Cu at 50 % coverage | 1.2 mm | 45.5 in-plane, 0.39 through | Rule of mixtures, computed in `pcb_effective_k()` |
| Bracket pad | Compliant silicone | 0.3 mm | 1.0 | |
| Camera–IMU bracket | Al 6061-T6, 60 × 20 × 2 mm | 2.0 mm | 167 | IMU end under the SoC, camera end at the device edge |
| Air gap | Air + gap radiation | ~6 mm | 0.044 | Ra ≪ 1708, so no internal convection |
| Bottom wall | Polycarbonate | 1.5 mm | 0.20 | Skin side |

**Boundary conditions**

* Top face: Lloyd–Moran natural convection (hot plate facing up, L = A/P) plus linearised radiation
  (ε = 0.9), recomputed until converged (Picard iteration).
* End walls: Churchill–Chu vertical-plate convection plus radiation.
* Front/back walls (outside the 2D slice): added back as plastic fins attached to the top and
  bottom walls (classic fin equation, adiabatic tip at mid-height).
* Bottom face: skin contact. Conductance h_skin = 35 W/m²K to a 37 °C core, from ~1 cm of
  perfused tissue (k ≈ 0.37) plus a strap contact resistance. Sanity check: with no device this
  gives bare skin ≈ 34 °C in 25 °C air, which is physiological.
* Limit: **43 °C on any outer surface** (prolonged-contact comfort limit used for wearables).

**Design variants**

| Key | Variant | What changes |
|---|---|---|
| a | Baseline | Plastic enclosure, no spreader, generic gap pad |
| b | Copper spreader | 0.5 mm Cu sheet under the top wall (100 × 80 mm) |
| c | Graphite spreader | 0.1 mm PGS (anisotropic: 700 in-plane / 15 through) |
| d | Thin high-k TIM | Thermal gel, 0.15 mm, 6 W/mK |
| e | Vented enclosure | Convective h × 2.5 over ±25 mm of the top wall above the SoC |
| f | **Recommended** | Two 100 µm PGS sheets (0.2 mm), generic gap pad, sealed |

---

## 3. Model assumptions and limitations

* **2D slice with depth homogenisation.** The solver models the x–y cross-section extruded over the
  80 mm depth. Parts that don't span the depth (12 mm die, 20 mm bracket) are mixed with
  air in proportion to the depth they occupy (`homogenize()`). This keeps their real
  through-thickness conductance and heat capacity. **What it misses:** heat from the die also has
  to spread in the depth direction. In 3D, the junction and the hotspot directly above the SoC
  will be somewhat **hotter** than the 2D values. Treat the 2D hotspot as optimistic and keep
  margin. A 3D check (FEA or IR measurement) is the first validation step.
* Steady natural convection in still air, top face facing up. Orientation, clothing, wind and
  sunlight all change h. A vertical orientation or a hat/sleeve covering is worse.
* Radiation is linearised to surroundings at room temperature. Internal gaps use a
  radiation-augmented air conductivity.
* Single heat source. Cameras, IMU, battery and PMIC dissipation are ignored, which is optimistic.
  No battery thermal mass either, so the real warm-up is slower.
* Constant properties (k at ~50 °C). Thick graphite stacks keep the in-plane k of one sheet, which
  is optimistic for thick single sheets.
* Vent enhancement (×2.5 on h) is an **assumption**, not a computation. It needs a test.
* Skin is a fixed conductance to a fixed core temperature, so there's no thermoregulation response.
* The bracket is assumed kinematically mounted (one end fixed, one sliding on a compliant pad), so
  it expands freely without thermal stress.

---

## 4. Physics implemented

1. **Resistance network** (`network1d.py`). Seven-node network (junction, top-plate inner face, top
   surface, PCB, bottom wall inner/outer, plus ambient and body), solved by nodal analysis.
   Spreading resistance into the top plate and the PCB uses the exact Fourier-series solution for
   a strip source on a layered, anisotropic plate. h is iterated with the same correlations as
   the 2D model, so the two models are independent apart from shared inputs.
2. **2D steady conduction** (`solver2d.py`). Finite-volume (conservative finite-difference) scheme on a
   non-uniform mesh with every material interface on a grid line. Harmonic-mean conductances
   between cells handle k jumps of 10⁴, and anisotropic kx/ky are supported. Convective, radiative,
   fin and skin boundaries are included, and the sparse system is solved directly
   (`scipy.sparse.linalg.spsolve`). The default mesh is ~14,000 cells.
3. **Transient warm-up** (`transient.py`). Backward-Euler time stepping (unconditionally stable) with
   step sizes from 0.2 s to 15 s, refreshing h as the surfaces warm. Runs 3 h from a cold start.
4. **Thermal expansion** (`expansion.py`). Takes the bracket's temperature field from the 2D solution.
   It computes axial growth u = ∫α(T−T_cal)dx, curvature κ = α·dT/dy, rotation θ = ∫κ dx, and bow
   w = ∫θ dx. Each is checked against the hand formulas ΔL = αLΔT and θ = αLΔT_through/h. The
   Timoshenko bimetal formula gives the "if rigidly bonded to the PCB" comparison.

---

## 5. Sanity checks (all 47 pass, see `results/sanity_checks.txt`)

| Check | Result |
|---|---|
| Steady energy balance (heat in = heat out), every variant | error < 1e-11 |
| Transient energy balance (generated = stored + lost) | error < 1e-10 |
| Transient ends at the steady solution | within 0.05 K after 3 h |
| Grid convergence (4 meshes, 3.7k → 220k cells) | finest two meshes differ by 0.04 K. Default mesh is within **0.19 K** of the Richardson-extrapolated value |
| 1D network vs 2D solver at junction, top surface, PCB, skin side | within **±12 %** of temperature rise for all variants (spreader variants within ±2 %). The worst case is the vented variant, because the network smears the vent h across the whole face |
| Maximum principle | the hottest cell is in the die, and nothing is colder than the room |
| Temperature ordering | Tj > PCB > skin side, Tj > peak surface > ambient |
| Correlation validity | top-face Ra ≈ 2e4, inside Lloyd–Moran's 1e4–1e7 range |
| Bracket integral vs hand calc | within 1 % |
| Spreader sweep is monotonic | yes, for Cu and PGS |
| Zero-power test | with the SoC off, the device sits between 25 and 37 °C, and the body's heat (0.56 W) all leaves through the air-facing faces |

### Non-physical results found and fixed during development

* **The front/back walls were missing.** A 2D slice silently drops two faces with ~25 % of the
  convective area. Ignoring them made the outcome depend on a modelling omission, so they're now
  modelled as fins (`solver2d.py`, `network1d.py`).
* **The "better TIM" result looked wrong, but it isn't.** A gel TIM *lowers* the sustainable
  power once a spreader is present. The gel sends more heat straight up into the hotspot and less
  down through the PCB. This was checked with both the 1D and 2D models and kept, because it is
  real physics and an important design lesson.
* **A kink in the transient surface curve.** For the first ~3 minutes the hottest surface is the
  skin side, warmed by the body. Then the top surface takes over. The kink is the maximum switching
  faces, not a numerical artefact.

---

## 6. Plots (in `results/`)

| # | File | What it shows | What to take away |
|---|---|---|---|
| 1 | `01_stackup_schematic.png` | Annotated cross-section (×3 vertical) + true-scale zoom + property table | The stack and where the heat starts |
| 2 | `02_resistance_network_baseline.png`, `02b_..._recommended.png` | The network with every R (K/W), heat flow (W) and node temperature, plus 2D values for comparison | Baseline: only **0.34 W** goes through the TIM, because spreading into plastic is 47 K/W. Most heat detours through the PCB and the air pocket. With PGS, 2.15 W goes straight up |
| 3 | `03_contours_baseline_vs_best.png` | 2D temperature fields, same colour scale (×2 vertical) | The baseline hotspot over the SoC disappears with graphite. The whole top runs evenly warm |
| 4 | `04_heat_flux_best.png` | Isotherms + heat-flux directions (true scale) | Heat runs sideways in the PGS and PCB, then up through the top wall. Only ~10 % goes into the body because the air gap insulates it |
| 5 | `05_variant_comparison.png` | Junction, peak surface, bracket ΔT, sustainable power for a–f | Spreaders give the big wins. TIM alone does nothing. Vents help less than spreading. No variant meets 43 °C at 3.5 W |
| 6 | `06_spreader_sweep.png` | Tj, peak surface and added mass vs spreader thickness, Cu vs PGS | Diminishing returns past ~0.2 mm. PGS matches copper at ~1/18 of the mass |
| 7 | `07_transient_warmup.png` | Warm-up curves, baseline vs recommended | The surface hits 43 °C after 3.4 min (baseline) vs **8.5 min** (recommended): the burst budget |
| 8 | `08_bracket_expansion.png` | Bracket temperature, axial growth, thermal bow | Growth (~35 µm) matters. Gradient bow (<1″) doesn't, *if* the mount is compliant. Rigid bonding would cost ~750″ |
| 9 | `09_grid_convergence.png` | Tj and peak surface vs mesh size | The answer is mesh-independent to ~0.2 K |

---

## 7. Results (3.5 W, 25 °C room)

| Variant | Junction (°C) | Peak surface (°C) | Skin side (°C) | Bracket ΔT (K) | Sustainable power (W) | Added mass (g) |
|---|---|---|---|---|---|---|
| a) Baseline | 59.4 | **55.2** | 39.3 | 3.16 | 1.83 | 0 |
| b) Cu 0.5 mm | 52.9 | 46.2 | 38.8 | 1.70 | **2.84** | 35.6 |
| c) PGS 0.1 mm | 53.8 | 47.3 | 38.8 | 1.92 | 2.66 | 0.7 |
| d) Thin high-k TIM | 58.4 | 55.0 | 39.1 | 3.16 | 1.85 | ≈0 |
| e) Vented | 55.7 | 49.8 | 38.8 | 2.80 | 2.34 | 0 |
| **f) 2 × PGS, sealed** | **53.2** | **46.5** | **38.8** | **1.77** | **2.79** | **1.4** |

"Sustainable power" is the highest continuous SoC power that keeps every outer surface ≤ 43 °C
(solved with a secant search). The skin-contact face stays below 40 °C in every variant. **The top
face is the binding constraint.**

**What it takes to run 3.5 W continuously** (all on top of f):

| Option | Top peak (°C) | Skin side (°C) | Sustainable power (W) | Added mass (g) |
|---|---|---|---|---|
| f alone | 46.5 | 38.8 | 2.79 | 1.4 |
| + thin high-k TIM | 47.5 | 38.4 | 2.63 ↓ | 1.2 |
| + conductive top shell (2 W/mK) | 46.3 | 38.5 | 2.82 | 1.4 |
| + skin-side path (PGS + gap filler) | 42.6 | 40.9 | 3.62 | 23.3 |
| + vents over SoC | 42.2 | 38.3 | 3.72 | 1.4 |
| + vents + skin path | 39.7 | 40.0 | 4.75 | 23.3 |
| f in a **35 °C** room | 54.1 | 40.2 | **1.30** | 1.4 |

---

## 8. Engineering recommendation

**Adopt variant f: two 100 µm pyrolytic graphite sheets laminated under the top shell, keep the
generic gap pad, keep the enclosure sealed, and run the SoC under a skin-temperature governor.**

* **Thermal.** Hotspot drops from 55.2 → 46.5 °C (−8.7 K), junction drops from 59.4 → 53.2 °C,
  and sustainable power rises from 1.83 → **2.79 W** (+52 %). That is 98 % of what 0.5 mm copper
  achieves (2.84 W).
* **Weight.** +1.4 g vs +35.6 g for copper, which matters on a head- or body-worn device.
* **Precision.** Bracket end-to-end gradient halves (3.16 → 1.77 K). Axial growth drops from 39.8 → 35.1 µm,
  and free-mount rotation is 3.0 µrad (0.63″).
* **Manufacturability.** Graphite with adhesive is a die-cut, pick-and-place laminate. The generic
  gap pad absorbs a ±0.2 mm stack tolerance that a 0.15 mm gel bond line can't. The gel's only
  benefit (−2.7 K junction) is on a non-binding constraint, and it costs 0.16 W of sustainable
  power.
* **Power budget.** At 3.5 W the sealed 120 × 80 × 14 mm envelope is **surface-area-limited**: the
  mean top-surface temperature is already ~44 °C, whatever spreads the heat. So the firmware
  budget is **2.8 W continuous, with 3.5 W bursts of up to 8.5 min from cold**. The budget drops to
  ~1.3 W in a 35 °C room, so the governor must read a skin/shell thermistor, not a timer.
* **If 3.5 W continuous is a hard product requirement**, the next lever is vents over the SoC
  (3.72 W, +0 g). This gives only 6 % margin, rests on an *assumed* ×2.5 h that must be measured,
  and costs sealing (sweat, dust, camera-window fogging). The alternative is a skin-side heat path
  (3.62 W), which costs ~23 g and raises skin temperature to ~41 °C. Growing the envelope is
  the third option.
* **Mounting.** Keep the camera–IMU bracket kinematically mounted on a compliant pad. If it were rigidly
  bonded to the FR-4, CTE mismatch would rotate the camera ~3.7 mrad (~750″), about 1000× the
  gradient effect. Calibrate extrinsics at operating temperature, or model them as a function of
  bracket temperature (~1.4 µm/K axial growth).

---

## 9. Talking points for the interview

* "Spreaders fix the peak, area fixes the average." Quote the numbers: the mean top surface stays
  at 43–45 °C in every sealed variant, while the peak moves from 55 to 46 °C.
* "Find the binding constraint before optimising." Tj has 40 K of margin, so a premium TIM is
  wasted. It even redirects heat into the hotspot.
* "In-plane conductance k·t is the spreader figure of merit." 0.2 mm PGS (k·t = 0.14 W/K)
  ≈ 0.5 mm Cu (0.19 W/K) once the plastic wall becomes the bottleneck.
* "The plastic shell is a resistor too." 1.5 mm PC at ~300 W/m² drops ~2.5 K by itself.
* "Validate cheap models against each other." The hand network agrees with the 2D solver within
  12 %, energy closes to 1e-11, and the mesh error is 0.2 K.
* "Thermal → precision is mostly a mounting problem." It's micrometres of growth and sub-arcsecond
  bow when the bracket can float, and hundreds of arcseconds when it can't.
