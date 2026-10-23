"""
expansion.py -- from temperature field to camera-IMU misalignment.

The camera and the IMU sit at opposite ends of a 60 mm aluminium bracket.
Their relative pose (the "extrinsic calibration") was measured in the factory
at T_CALIBRATION. Heat from the SoC changes the bracket in two ways:

1. Axial growth. Every slice dx grows by alpha*(T - T_cal)*dx, so the
   camera-to-IMU distance changes by
        dL = integral of alpha*(T_mean(x) - T_cal) dx      (~ alpha*L*dT)
   This shifts the translation part of the extrinsics.

2. Thermal bow. If the side of the bracket facing the PCB is hotter than the
   far side, that fibre grows more and the beam curls (bimetal effect with a
   single metal). For a linear through-thickness gradient, Euler-Bernoulli
   beam theory gives the curvature
        kappa(x) = alpha * dT/dy                        [1/m]
   and the relative rotation of the two ends is
        d_theta = integral of kappa dx
   Hand check: the top fibre grows alpha*L*dT_through more than the bottom
   fibre, spread over the thickness h, so d_theta ~ alpha*L*dT_through / h.
   This rotation tilts the camera relative to the IMU -- a direct error in
   the rotation part of the extrinsics.

Assumption: the bracket is kinematically mounted (one end fixed, the other on
a compliant pad), so it expands freely and carries no thermal stress. If it
were rigidly bonded to the PCB, the CTE mismatch between aluminium (23.6 ppm/K)
and FR-4 (~16 ppm/K) would dominate instead; we report that number too.
"""

import numpy as np

import device as dev
from solver2d import bracket_field

ARCSEC_PER_RAD = 180 / np.pi * 3600


def _cumtrapz(y, x):
    """Cumulative trapezoidal integral starting at 0 (numpy-only)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def bimetal_curvature(d_alpha, d_temp, t1, t2, e1, e2):
    """Curvature of two bonded strips heated uniformly (Timoshenko, 1925).

        kappa = 6 d_alpha dT (1+m)^2 / ( h [3(1+m)^2 + (1+m n)(m^2 + 1/(m n))] )

    with m = t1/t2, n = E1/E2, h = t1 + t2. This is what happens if the
    aluminium bracket is glued rigidly along its length to the FR-4 board.
    """
    m, n, h = t1 / t2, e1 / e2, t1 + t2
    return (6 * d_alpha * d_temp * (1 + m) ** 2
            / (h * (3 * (1 + m) ** 2 + (1 + m * n) * (m ** 2 + 1 / (m * n)))))


def bracket_deformation(sol, t_cal=dev.T_CALIBRATION):
    """Axial displacement, bow and end-to-end rotation of the bracket.

    Returns a dict with arrays along the bracket (x measured from the IMU
    end) and scalar summary values.
    """
    xc, yc, Tb = bracket_field(sol)
    alpha = dev.AL6061.cte
    h = dev.BRACKET_THICK
    x = np.concatenate([[dev.BRACKET_X0], xc, [dev.BRACKET_X1]]) - dev.BRACKET_X0

    # Least-squares slope dT/dy and mean temperature in each bracket column.
    yc0 = yc - yc.mean()
    grad = (yc0 @ Tb) / np.sum(yc0 ** 2)          # K/m, +ve = top hotter
    t_mean = Tb.mean(axis=0)
    # extend to the bracket ends (constant extrapolation over half a cell)
    grad = np.concatenate([[grad[0]], grad, [grad[-1]]])
    t_mean = np.concatenate([[t_mean[0]], t_mean, [t_mean[-1]]])

    strain = alpha * (t_mean - t_cal)
    u = _cumtrapz(strain, x)                      # axial displacement, m
    kappa = alpha * grad                          # curvature, 1/m
    theta = _cumtrapz(kappa, x)                   # slope, rad
    w = _cumtrapz(theta, x)                       # deflection, m
    w -= w[-1] * x / x[-1]                        # supports at both ends: w(0)=w(L)=0

    dT_through = float(np.average(grad, weights=np.gradient(x)) * h)
    d_theta = float(theta[-1])
    hand = alpha * dev.BRACKET_LENGTH * dT_through / h
    kappa_bonded = bimetal_curvature(dev.AL6061.cte - dev.PCB.cte, t_mean.mean() - t_cal,
                                     h, dev.PCB_THICK, dev.E_BRACKET, dev.E_PCB)
    return {
        "x": x, "T_mean": t_mean, "u": u, "w": w, "theta": theta,
        "dT_ends": float(t_mean[0] - t_mean[-1]),      # IMU end minus camera end
        "dT_max": float(Tb.max() - Tb.min()),
        "dT_through": dT_through,
        "dL_um": float(u[-1] * 1e6),
        "dL_hand_um": float(alpha * dev.BRACKET_LENGTH * (t_mean.mean() - t_cal) * 1e6),
        "rotation_urad": d_theta * 1e6,
        "rotation_arcsec": d_theta * ARCSEC_PER_RAD,
        "rotation_hand_urad": hand * 1e6,
        "bow_um": float(np.max(np.abs(w)) * 1e6),
        "bonded_rotation_urad": float(kappa_bonded * dev.BRACKET_LENGTH * 1e6),
        "bonded_rotation_arcsec": float(kappa_bonded * dev.BRACKET_LENGTH * ARCSEC_PER_RAD),
        "cte_mismatch_um": float((dev.AL6061.cte - dev.PCB.cte) * dev.BRACKET_LENGTH
                                 * (t_mean.mean() - t_cal) * 1e6),
    }
