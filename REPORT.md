# Engineering Summary: Thermal Path of the Wearable Sensor Module

**Device:** 120 × 80 × 14 mm sealed PC enclosure, 3.5 W SoC (12 × 12 mm), 60 mm Al camera–IMU
bracket, worn on the body in a 25 °C room. **Requirement:** every outer surface ≤ 43 °C.
**Method:** 1D resistance network + 2D finite-volume model (steady and transient) + bracket
expansion model. 47 automatic checks pass: energy balance to 1e-11, mesh error 0.2 K, and
1D vs 2D agreement within 12 %.

## Recommendation

**Two 100 µm pyrolytic graphite sheets under the top shell (variant f), generic gap pad kept,
enclosure sealed, SoC run under a shell-temperature governor.**

| Metric | Baseline | Recommended (f) | Cu 0.5 mm (b) |
|---|---|---|---|
| Junction temperature | 59.4 °C | 53.2 °C | 52.9 °C |
| Peak outer surface @ 3.5 W | 55.2 °C | 46.5 °C | 46.2 °C |
| Skin-contact face peak | 39.3 °C | 38.8 °C | 38.8 °C |
| Sustainable power (all surfaces ≤ 43 °C) | 1.83 W | **2.79 W** | 2.84 W |
| Time to 43 °C at 3.5 W, cold start | 3.4 min | **8.5 min** | — |
| Bracket end-to-end ΔT | 3.16 K | 1.77 K | 1.70 K |
| Added mass | — | **1.4 g** | 35.6 g |

**Operating budget:** 2.8 W continuous, with 3.5 W bursts of ≤ 8.5 min from cold. In a 35 °C room the
continuous budget falls to 1.3 W, so the governor must close the loop on a measured shell
temperature. If 3.5 W *continuous* is mandatory, adding vents over the SoC reaches 3.72 W, but
that has only 6 % margin, rests on an assumed vent h, and gives up sealing. The other options
are a skin-side conduction path (3.62 W, +23 g, skin ~41 °C) or a larger envelope.

## The key trade-off

| | Precision | Weight | Thermal | Cost / manufacturing |
|---|---|---|---|---|
| **2 × PGS (rec.)** | Gradient halved | +1.4 g | 98 % of Cu | Die-cut laminate. Higher material cost/cm² than Cu but no forming |
| Cu 0.5 mm | Best gradient (1.70 K) | **+35.6 g** | Best (2.84 W) | Cheap sheet, but heavy. Needs insulation from other parts |
| Thin gel TIM | Smaller gradient | ≈0 | **Worse** surface (−0.16 W budget) | Dispensing, tight tolerance stack |
| Vents | Neutral | 0 | +0.9 W (assumed h) | Lost IP rating, sweat/dust, window fogging |
| Skin path | Neutral | +23 g | +0.8 W | Warmer skin (~41 °C), more parts |

The surface temperature, not the junction, is the binding constraint (Tj has ~40 K of margin).
So the right spend is on *spreading* (cheap in mass with graphite), not on the TIM. The
remaining shortfall at 3.5 W comes from envelope area, which no internal part can fix. It is a
product decision: power budget, venting, or size.

**Sensor accuracy.** With a compliant (kinematic) bracket mount, thermal bow rotates the camera
relative to the IMU by only ~3 µrad (0.6″). The relevant effect is axial growth of ~35 µm
(≈1.4 µm/K), which a temperature-indexed extrinsic calibration removes. Rigidly bonding the Al
bracket to FR-4 would instead cause ~3.7 mrad (~750″) of bimetal rotation. **Bracket mounting
is the precision decision. The thermal design only sets its magnitude.**

## What I would measure on a prototype

1. **Shell temperature map.** IR camera (with the surface painted or taped to a known ε ≈ 0.95), plus
   thermocouples at the hotspot and four edges, at 1.0/2.0/3.5 W steady state in a still-air box.
   Compare peak and mean against the model. 3D die spreading is expected to make the real hotspot
   hotter than this 2D model.
2. **Junction.** SoC on-die thermal sensor vs. power, to extract the junction-to-shell
   resistance and check the TIM bond line (also check bond-line thickness by cross-section or CT).
3. **Warm-up time constant.** Step from 0 → 3.5 W, log the surface every second, and fit the time to
   43 °C (model: 8.5 min). The battery's thermal mass will lengthen this.
4. **Convection and radiation h.** A heated-plate replica of the enclosure, measured in orientations
   (face up, vertical, under a sleeve). This is critical before trusting the vent ×2.5 assumption.
5. **Skin-side boundary.** A skin-simulant pad or a volunteer study (thermocouple under the strap).
   This gives contact temperature and the effective h_skin (model: 35 W/m²K).
6. **Ambient sweep.** 25/30/35 °C chamber tests to confirm the governor budget (model: 2.8 → 1.3 W).
7. **Bracket.** Thermocouples at both bracket ends (model ΔT 1.8 K). Measure camera–IMU extrinsics
   by checkerboard plus IMU rotation calibration, cold vs. at temperature. Confirm rotation is
   < 10 µrad and that translation drift follows ~1.4 µm/K.
