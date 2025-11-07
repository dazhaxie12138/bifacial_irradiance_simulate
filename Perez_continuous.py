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

    I0 = 1353 * (1 + 0.033 * np.cos(np.radians(360 * (julian_days - 2) / 365)))
    delta = DHI / (I0 * np.sin(solar_altitude))
    epsilon = (((DHI + DNI) / DHI + 1.041 * (pi / 2 - solar_altitude) ** 3) /
               (1 + 1.041 * (pi / 2 - solar_altitude) ** 3))

    # convert zeta parameter (continunes convert)
    zeta = 1 - 1 / epsilon

    t = np.array([0.000, 0.000, 0.000, 0.061, 0.187, 0.333, 0.487, 0.643, 0.778, 0.839, 1.000, 1.000, 1.000])

    # spline coefficient
    coeffs_dict = {'F11': np.array([-0.053, -0.008, 0.131, 0.328, 0.557, 0.861,
                                    1.212, 1.099, 0.544, 0.544, 0.000, 0.000, 0.000]),
                   'F12': np.array([0.529, 0.588, 0.770, 0.471, 0.241, -0.323,
                                    -1.239, -1.847, 0.157, 0.157, 0.000, 0.000, 0.000]),
                   'F13': np.array([-0.028, -0.062, -0.167, -0.216, -0.300, -0.355,
                                    -0.444, -0.365, -0.213, -0.213, 0.000, 0.000, 0.000]),
                   'F21': np.array([-0.071, -0.060, -0.026, 0.069, 0.086, 0.240,
                                    0.305, 0.275, 0.118, 0.118, 0.000, 0.000, 0.000]),
                   'F22': np.array([0.061, 0.072, 0.106, -0.105, -0.085, -0.467,
                                    -0.797, -1.132, -1.455, -1.455, 0.000, 0.000, 0.000]),
                   'F23': np.array([-0.019, -0.022, -0.032, -0.028, -0.012, -0.008,
                                    0.047, 0.124, 0.292, 0.292, 0.000, 0.000, 0.000])}

    def vectorized_spline_interp(zeta, t, coeff):
        indices = np.searchsorted(t, zeta, side='right') - 1

        # insured the index is located valid range [0, len(t)-3]
        indices = np.clip(indices, 0, len(t) - 4)

        # Calculate the offset of each zeta value relative to the start of the interval
        offsets = zeta - t[indices]

        result = (coeff[indices] + coeff[indices + 1] * offsets + coeff[indices + 2] * offsets ** 2)

        # Handle edge cases (zeta out of scope)
        out_of_range = (zeta < t[0]) | (zeta > t[-1])
        result[out_of_range] = 0.0

        return result

    F11 = vectorized_spline_interp(zeta, t, coeffs_dict['F11'])
    F12 = vectorized_spline_interp(zeta, t, coeffs_dict['F12'])
    F13 = vectorized_spline_interp(zeta, t, coeffs_dict['F13'])
    F21 = vectorized_spline_interp(zeta, t, coeffs_dict['F21'])
    F22 = vectorized_spline_interp(zeta, t, coeffs_dict['F22'])
    F23 = vectorized_spline_interp(zeta, t, coeffs_dict['F23'])

    # Calculate final coefficients
    F1 = F11 + delta * F12 + (pi / 2 - solar_altitude) * F13
    F2 = F21 + delta * F22 + (pi / 2 - solar_altitude) * F23

    # 应Apply physical constraints
    F1 = np.clip(F1, 0, 0.9)


    return F1, F2

