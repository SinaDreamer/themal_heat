"""
convection.py -- heat transfer coefficients for the outer surfaces.

A sealed wearable has no fan, so the outside of the enclosure loses heat by
  1. natural (buoyancy-driven) convection to the room air, and
  2. thermal radiation to the surroundings.
Both are comparable in size (~5-8 W/m^2/K each), which is why emissivity
matters as much as the shape of the device.

The coefficients depend on the surface temperature, so the solvers iterate:
guess h -> solve temperatures -> update h -> repeat until nothing changes.
"""

import numpy as np

from device import EMISSIVITY, G_ACCEL, KELVIN, SIGMA

# Dry air at 1 atm [Incropera Table A.4], columns: T (K), nu (m^2/s), k (W/m/K), alpha (m^2/s)
_AIR_TABLE = np.array([
    [300.0, 15.89e-6, 0.0263, 22.5e-6],
    [350.0, 20.92e-6, 0.0300, 29.9e-6],
])


def air_properties(t_film_c: float):
    """Air properties at the film temperature (C), linearly interpolated.

    Returns (nu, k, alpha, beta) where beta = 1/T_film is the ideal-gas
    volumetric expansion coefficient.
    """
    t = t_film_c + KELVIN
    nu = np.interp(t, _AIR_TABLE[:, 0], _AIR_TABLE[:, 1])
    k = np.interp(t, _AIR_TABLE[:, 0], _AIR_TABLE[:, 2])
    alpha = np.interp(t, _AIR_TABLE[:, 0], _AIR_TABLE[:, 3])
    return nu, k, alpha, 1.0 / t


def rayleigh(t_surface: float, t_ambient: float, length: float) -> float:
    """Rayleigh number Ra = g*beta*dT*L^3 / (nu*alpha) for natural convection."""
    nu, _, alpha, beta = air_properties(0.5 * (t_surface + t_ambient))
    dt = max(abs(t_surface - t_ambient), 1e-3)
    return G_ACCEL * beta * dt * length ** 3 / (nu * alpha)


def h_horizontal_up(t_surface: float, t_ambient: float, area: float, perimeter: float) -> float:
    """Natural convection from a hot plate facing up (the outer top face).

    Lloyd & Moran correlation, Nu = 0.54 Ra^(1/4) for 1e4 < Ra < 1e7, with the
    characteristic length L = A/P [Incropera eq. 9.30].
    """
    length = area / perimeter
    ra = rayleigh(t_surface, t_ambient, length)
    _, k, _, _ = air_properties(0.5 * (t_surface + t_ambient))
    nu_l = 0.54 * ra ** 0.25
    return nu_l * k / length


def h_vertical(t_surface: float, t_ambient: float, height: float, pr: float = 0.707) -> float:
    """Natural convection from a vertical plate (the end walls).

    Churchill & Chu correlation, valid for all Ra [Incropera eq. 9.26]:
        Nu = {0.825 + 0.387 Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)}^2
    """
    ra = rayleigh(t_surface, t_ambient, height)
    _, k, _, _ = air_properties(0.5 * (t_surface + t_ambient))
    nu_l = (0.825 + 0.387 * ra ** (1 / 6) / (1 + (0.492 / pr) ** (9 / 16)) ** (8 / 27)) ** 2
    return nu_l * k / height


def h_radiation(t_surface, t_ambient: float, emissivity: float = EMISSIVITY):
    """Linearised radiation coefficient to large surroundings at ambient.

    q = eps*sigma*(Ts^4 - Tinf^4) = h_rad*(Ts - Tinf) with
    h_rad = eps*sigma*(Ts + Tinf)*(Ts^2 + Tinf^2)  [Incropera eq. 1.9].
    Works element-wise on numpy arrays (local surface temperatures).
    """
    ts = np.asarray(t_surface) + KELVIN
    ta = t_ambient + KELVIN
    return emissivity * SIGMA * (ts + ta) * (ts ** 2 + ta ** 2)
