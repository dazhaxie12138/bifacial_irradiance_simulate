"""
this code used to calculated the circumsolar and horizontal brightness coefficient, 
from the article "A continuous form of the Perez diffusesky model for forward and reverse transposition
"""

import pandas as pd
import numpy as np
from math import pi


def PerezDriesseContinuous(DNI, DHI, solar_altitude, start_date=None, end_date=None, time_step=None):
    """
    parameter:
    DNI: direct normal irradiance [W/m²]
    DHI: horizontal diffuse irradiance [W/m²]
    solar_altitude:  [radiance]
    return:
    F1, F2:
    """
    if isinstance(DNI, pd.Series):
        time_series = DNI.index.to_series()
        julian_days = time_series.dt.dayofyear
    else:
        time_series = pd.date_range(start=start_date, end=end_date, freq=f'{time_step}min').to_series()
        julian_days = time_series.dt.dayofyear[:-1]

    # 天顶角
    Z = (pi / 2 - solar_altitude)

    # 地外辐照
    I0 = 1353 * (1 + 0.033 * np.cos(np.radians(360 * (julian_days - 2) / 365)))

    # 大气质量
    am = (1.0 / (np.cos(Z) + 0.50572 * ((6.07995 + (90 - np.degrees(Z))) ** - 1.6364)))
    max_am = (1.0 / (np.cos(pi/2) + 0.50572 * (6.07995 ** - 1.6364)))
    am = np.where(Z >= pi/2, max_am, am)

    # delta and epsilon
    delta = DHI / (I0 / am)
    epsilon = (((DHI + DNI) / DHI + 1.041 * Z ** 3) / (1 + 1.041 * Z ** 3))

    # --- 4. zeta ---
    zeta = 1 - 1 / epsilon

    # parameter from Table 2
    t = np.array([0.000, 0.000, 0.000, 0.061, 0.187, 0.333,
                  0.487, 0.643, 0.778, 0.839, 1.000, 1.000, 1.000])

    c11 = np.array([-0.053, -0.008, 0.131, 0.328, 0.557, 0.861,
                    1.212, 1.099, 0.544, 0.544, 0.000, 0.000, 0.000])
    c12 = np.array([0.529, 0.588, 0.770, 0.471, 0.241, -0.323,
                    -1.239, -1.847, 0.157, 0.157, 0.000, 0.000, 0.000])
    c13 = np.array([-0.028, -0.062, -0.167, -0.216, -0.300, -0.355,
                    -0.444, -0.365, -0.213, -0.213, 0.000, 0.000, 0.000])

    c21 = np.array([-0.071, -0.060, -0.026, 0.069, 0.086, 0.240,
                    0.305, 0.275, 0.118, 0.118, 0.000, 0.000, 0.000])
    c22 = np.array([0.061, 0.072, 0.106, -0.105, -0.085, -0.467,
                    -0.797, -1.132, -1.455, -1.455, 0.000, 0.000, 0.000])
    c23 = np.array([-0.019, -0.022, -0.032, -0.028, -0.012, -0.008,
                    0.047, 0.124, 0.292, 0.292, 0.000, 0.000, 0.000])

    # --- 6. 真正的二次 B-spline ---
    k = 2

    spline_F11 = BSpline(t, c11, k, extrapolate=False)
    spline_F12 = BSpline(t, c12, k, extrapolate=False)
    spline_F13 = BSpline(t, c13, k, extrapolate=False)

    spline_F21 = BSpline(t, c21, k, extrapolate=False)
    spline_F22 = BSpline(t, c22, k, extrapolate=False)
    spline_F23 = BSpline(t, c23, k, extrapolate=False)

    # zeta 限制在 0~1 避免越界
    zeta_clamp = np.clip(zeta, 0.0, 1.0)

    F11 = spline_F11(zeta_clamp)
    F12 = spline_F12(zeta_clamp)
    F13 = spline_F13(zeta_clamp)

    F21 = spline_F21(zeta_clamp)
    F22 = spline_F22(zeta_clamp)
    F23 = spline_F23(zeta_clamp)

    F1 = F11 + delta * F12 + Z * F13
    F2 = F21 + delta * F22 + Z * F23

    # F1 物理限制
    F1 = np.clip(F1, 0, 0.9)

    return F1, F2

