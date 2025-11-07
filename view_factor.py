"""
Author Zexing Deng
time 2025-11-17
This file is used to calculate various view factors involved in irradiance simulate.
"""

import numpy as np
from math import sin, radians, pi
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)


def cal_relative_parameter(D, A, h, beta):
    """ Calculate PV array installation characteristic parameters """
    hu = h + 0.5 * A * np.sin(beta)
    hb = h - 0.5 * A * np.sin(beta)
    lf = hu * (D + A * np.cos(beta)) / (A * np.sin(beta)) - A * np.cos(beta)
    lr = hu * (D - A * np.cos(beta)) / (A * np.sin(beta)) + A * np.cos(beta)
    le = hb / np.tan(beta)
    return hu, hb, lf, lr, le


def cal_F_gnd_sky(x, dx, D, A, h, beta, N):
    """
    Calculate the angle coefficient of the ground facing the sky within the field of view from the front or back of the specified photovoltaic array
    
    :param x: A form of np.arange(X1,X2,dx) where X1 is the starting position of the ground and X2 is the ending position of the ground
    :param dx: Micro-length ground segment
    :param D: row spacing
    :param A: height of the PV array
    :param h: height of the PV installation height
    :param beta: tilt angle
    :param N: Total number of rows of PV field
    :return: The view factor of each dx to the sky from X1 to X2
    """
    hu, hb, lf, lr, le = cal_relative_parameter(D, A, h, beta)
    Ns = np.arange(1, N)[:, np.newaxis]

    uncrossed_l1 = np.where(x >= (Ns - 1) * D - le,
                            np.sqrt(hu ** 2 + (x - (Ns - 1) * D - A * np.cos(beta)) ** 2),
                            np.sqrt(hu ** 2 + (hu * (x - (Ns - 1) * D) / hb) ** 2))

    uncrossed_l2 = np.where(x + dx >= Ns * D - le,
                            np.sqrt(hu ** 2 + (hu * (x + dx - Ns * D) / hb) ** 2),
                            np.sqrt(hu ** 2 + (x + dx - Ns * D - A * np.cos(beta)) ** 2))

    # 计算 cross_l1 和 cross_l2
    cross_l1 = np.where(x >= (Ns - 1) * D - le,
                        np.sqrt(hu ** 2 + (x + dx - (Ns - 1) * D - A * np.cos(beta)) ** 2),
                        np.sqrt(hu ** 2 + (hu * (x - (Ns - 1) * D) / hb + dx) ** 2))

    cross_l2 = np.where(x + dx >= Ns * D - le,
                        np.sqrt(hu ** 2 + (hu * (x + dx - Ns * D) / hb - dx) ** 2),
                        np.sqrt(hu ** 2 + (x - Ns * D - A * np.cos(beta)) ** 2))

    F = (cross_l1 + cross_l2 - uncrossed_l1 - uncrossed_l2) / (2 * dx)
    F[F < 0] = 0
    F_dx_middle_sky = np.nansum(F, axis=0)

    F_dx_front_sky = 0.5 + np.where(x + dx < -le,
                                    np.sqrt(hu ** 2 + (x - A * np.cos(beta)) ** 2) -
                                    np.sqrt(hu ** 2 + (x + dx - A * np.cos(beta)) ** 2),
                                    np.sqrt(hu ** 2 + (hu * (x + dx) / hb - dx) ** 2) -
                                    np.sqrt(hu ** 2 + (hu * (x + dx) / hb) ** 2)) / (2 * dx)
    F_dx_rear_sky = 0.5 + np.where(x <= (N - 1) * D - le,
                                   np.sqrt(hu ** 2 + (hu * (N * D - D - x) / hb - dx) ** 2) -
                                   np.sqrt(hu ** 2 + (hu * (N * D - D - x) / hb) ** 2),
                                   np.sqrt(hu ** 2 + (x + dx - N * D + D - A * np.cos(beta)) ** 2) -
                                   np.sqrt(hu ** 2 + (x - N * D + D - A * np.cos(beta)) ** 2)) / (2 * dx)
    F_dx_sky = np.nansum([F_dx_rear_sky, F_dx_front_sky, F_dx_middle_sky], axis=0)

    return F_dx_sky


def cal_F_pv_front_dx_sky(n, N, D, A, h, beta, altitude):
    """
    :param n:
    :param N:
    :param D:
    :param A:
    :param h:
    :param beta:
    :return: The proportion of diffuse reflected radiation received by the front of the photovoltaic array to DHI
    """

    hu, hb, lf, lr, le = cal_relative_parameter(D, A, h, beta)

    X1 = np.maximum((n - 1) * D - lf, -D)
    X2 = np.maximum((n - 1) * D - le, -D)
    Ng = np.ceil((X2 - X1) / D * 100)

    if isinstance(beta, np.ndarray):
        F_pvf_beneath_array = []
        for i in range(len(beta)):
            if X1[i] == X2[i] or altitude[i] < 0:
                view_factor_shaded = 0
            else:
                dx = (X2 - X1) / Ng
                x = np.arange(X1[i], X2[i], dx[i])
                F_dx_sky = cal_F_gnd_sky(x, dx[i], D, A, h, beta[i], N)
                F_pv_front_dx = (np.sqrt(hb[i] ** 2 + ((n - 1)[i] * D - x) ** 2) +
                                 np.sqrt(hu[i] ** 2 + ((n - 1)[i] * D + A * np.cos(beta[i]) - x - dx[i]) ** 2) -
                                 np.sqrt(hb[i] ** 2 + ((n - 1)[i] * D - x - dx[i]) ** 2) -
                                 np.sqrt(hu[i] ** 2 + ((n - 1)[i] * D + A * np.cos(beta[i]) - x) ** 2)) / (2 * A)
                view_factor_shaded = np.sum(F_pv_front_dx * F_dx_sky)
            F_pvf_beneath_array.append(view_factor_shaded)

        F_pvf_beneath_array = np.array(F_pvf_beneath_array)
    else:
        if X1 == X2:
            F_pvf_beneath_array = 0
        else:
            dx = (X2 - X1) / Ng
            x = np.arange(X1, X2, dx)
            F_dx_sky = cal_F_gnd_sky(x, dx, D, A, h, beta, N)
            F_pv_front_dx = (np.sqrt(hb ** 2 + ((n - 1) * D - x) ** 2) +
                             np.sqrt(hu ** 2 + ((n - 1) * D + A * np.cos(beta) - x - dx) ** 2) -
                             np.sqrt(hb ** 2 + ((n - 1) * D - x - dx) ** 2) -
                             np.sqrt(hu ** 2 + ((n - 1) * D + A * np.cos(beta) - x) ** 2)) / (2 * A)
            F_pvf_beneath_array = np.sum(F_pv_front_dx * F_dx_sky)

    l4 = np.minimum(lf, n * D)
    l5 = np.minimum(le, n * D)
    F_pvf_less_D = (np.sqrt(hb ** 2 + l4 ** 2) + np.sqrt(hu ** 2 + (l5 + A * np.cos(beta)) ** 2) -
                    np.sqrt(hb ** 2 + l5 ** 2) - np.sqrt(hu ** 2 + (l4 + A * np.cos(beta)) ** 2)) / (2 * A)

    F_pv_front_gnd = np.where(n == 1, (1 - np.cos(beta)) / 2,
                              (A + np.sqrt(lf ** 2 + hb ** 2) - np.sqrt((lf + A * np.cos(beta)) ** 2 + hu ** 2)) /
                              (2 * A))
    F_pv_dx_sky = F_pvf_beneath_array + F_pv_front_gnd - F_pvf_less_D

    return F_pv_dx_sky


def cal_F_pv_rear_dx_sky(n, N, D, A, h, beta, altitude):
    """
    :param n:
    :param N:
    :param D:
    :param A:
    :param h:
    :param beta:
    :return: The proportion of diffuse reflected radiation received by the backside of the photovoltaic array to DHI
    """

    hu, hb, lf, lr, le = cal_relative_parameter(D, A, h, beta)
    X1 = np.maximum((n - 1) * D - le, -D)
    X2 = np.minimum((n - 1) * D + lr, N * D)
    Ng = np.ceil((X2 - X1) / D * 100)

    if isinstance(beta, np.ndarray):
        F_pvr_beneath_array = []
        for i in range(len(beta)):
            if altitude[i] < 0:
                view_factor_shaded = 0
            else:
                dx = (X2 - X1) / Ng
                x = np.arange(X1[i], X2[i], dx[i])
                F_dx_sky = cal_F_gnd_sky(x, dx[i], D, A, h, beta[i], N)
                F_pv_rear_dx = (np.sqrt(hb[i] ** 2 + ((n - 1)[i] * D - x - dx[i]) ** 2) +
                                np.sqrt(hu[i] ** 2 + ((n - 1)[i] * D + A * np.cos(beta[i]) - x) ** 2) -
                                np.sqrt(hb[i] ** 2 + ((n - 1)[i] * D - x) ** 2) -
                                np.sqrt(hu[i] ** 2 + ((n - 1)[i] * D + A * np.cos(beta[i]) - x - dx[i]) ** 2)) / (2 * A)
                view_factor_shaded = np.sum(F_pv_rear_dx * F_dx_sky)

            F_pvr_beneath_array.append(view_factor_shaded)
        F_pvr_beneath_array = np.array(F_pvr_beneath_array)
    else:
        dx = (X2 - X1) / Ng
        x = np.arange(X1, X2, dx)
        F_dx_sky = cal_F_gnd_sky(x, dx, D, A, h, beta, N)
        F_pv_rear_dx = (np.sqrt(hb ** 2 + ((n - 1) * D - x - dx) ** 2) +
                        np.sqrt(hu ** 2 + ((n - 1) * D + A * np.cos(beta) - x) ** 2) -
                        np.sqrt(hb ** 2 + ((n - 1) * D - x) ** 2) -
                        np.sqrt(hu ** 2 + ((n - 1) * D + A * np.cos(beta) - x - dx) ** 2)) / (2 * A)
        F_pvr_beneath_array = np.sum(F_pv_rear_dx * F_dx_sky)

    # 漫射反射辐照计算
    l4 = np.minimum(le, n * D)
    l5 = np.minimum(lr, (N - n + 1) * D)
    F_pvr_less_D = (np.sqrt(hb ** 2 + l5 ** 2) +
                    np.sqrt(hu ** 2 + (l4 + A * np.cos(beta)) ** 2) -
                    np.sqrt(hb ** 2 + l4 ** 2) -
                    np.sqrt(hu ** 2 + (l5 - A * np.cos(beta)) ** 2)) / (2 * A)
    F_pv_rear_gnd = np.where(n < N,
                             (A + np.sqrt(hb ** 2 + lr ** 2) - np.sqrt(hu ** 2 + (lr - A * np.cos(beta)) ** 2)) /
                             (2 * A),
                             (1 + np.cos(beta)) / 2)
    F_pv_dx_sky = F_pvr_beneath_array + F_pv_rear_gnd - F_pvr_less_D

    return F_pv_dx_sky


def cal_F_PV_nsgnd_front(n, N, D, A, h, beta, azimuth_PV, solar_azimuth, solar_altitude):
    hu, hb, lf, lr, le = cal_relative_parameter(D, A, h, beta)
    Sa = A * np.cos(beta) + A * np.sin(beta) * np.cos(solar_azimuth - azimuth_PV) / np.tan(solar_altitude)
    Sb = np.where(Sa > 0, hb * np.cos(solar_azimuth - azimuth_PV) / np.tan(solar_altitude),
                  A * np.cos(beta) + hu * np.cos(solar_azimuth - azimuth_PV) / np.tan(solar_altitude))

    # front head VF
    lo = np.ceil((lf + Sb) / D) * D - (lf + Sb)
    l1 = np.minimum(lf + lo - np.abs(Sa), lf)
    lf = np.where(n == 1, float('inf'), lf)
    F_pv_front_f_nsgnd = np.where(lf > (n - 1) * D - Sb,
                                  np.where(n == 1,
                                           np.sqrt(hu ** 2 + (A * np.cos(beta) - Sb) ** 2) -
                                           np.sqrt(hb ** 2 + Sb ** 2) - A * np.cos(beta),
                                           np.sqrt(hb ** 2 + lf ** 2) +
                                           np.sqrt(hu ** 2 + ((n - 1) * D - Sb + A * np.cos(beta)) ** 2) -
                                           np.sqrt(hb ** 2 + ((n - 1) * D - Sb) ** 2) -
                                           np.sqrt(hu ** 2 + (lf + A * np.cos(beta)) ** 2)),
                                           np.where(np.abs(Sa) < D,
                                                    np.sqrt(hb ** 2 + l1 ** 2) +
                                                    np.sqrt(hu ** 2 + (lf + lo + A * np.cos(beta) - D) ** 2) -
                                                    np.sqrt(hb ** 2 + (lf + lo - D) ** 2) -
                                                    np.sqrt(hu ** 2 + (l1 + A * np.cos(beta)) ** 2), 0)) / (2 * A)

    # front tail VF
    lp = np.ceil((le + Sb) / D) * D - (le + Sb)
    l2 = np.minimum(le + lp - np.abs(Sa), le)
    F_pv_front_r_nsgnd = (np.where(le <= -((N - n) * D + np.abs(Sa) + Sb),
                                  A + np.sqrt(hb ** 2 + ((N - n) * D + np.abs(Sa) + Sb) ** 2) -
                                  np.sqrt(hu ** 2 + ((N - n) * D + np.abs(Sa) + Sb - A * np.cos(beta)) ** 2),
                                  np.where(np.logical_and(le < (n - 1) * D - np.abs(Sa) - Sb, np.abs(Sa) < D),
                                           np.sqrt(hb ** 2 + (le + lp - np.abs(Sa)) ** 2) +
                                           np.sqrt(hu ** 2 + (l2 + A * np.cos(beta)) ** 2) -
                                           np.sqrt(hb ** 2 + l2 ** 2) -
                                           np.sqrt(hu ** 2 + (le + lp - np.abs(Sa) + A * np.cos(beta)) ** 2), 0)) /
                                           (2 * A))

    # front middle VF
    M = np.minimum(np.floor((lf + Sb) / D), n - 1) - np.maximum(np.minimum(np.ceil((le + Sb) / D), n - 1), 0)
    maxM = M.max()
    index_m = np.arange(1, maxM + 1)
    Sa_b = np.abs(Sa).values[:, None]
    l3_b = np.where(lf < (n - 1) * D - Sb, lf + lo - D, (n - 1) * D - Sb)[:, None]
    valid = ((M >= 1) & (np.abs(Sa) < D)).values
    mask = (index_m[None, :] <= M[:, None]) & valid[:, None]

    if isinstance(beta, np.ndarray):
        beta_bed = beta[:, None]
        hb_b = hb[:, None]
        hu_b = hu[:, None]
        F_m = (np.sqrt(hb_b ** 2 + (l3_b - (index_m - 1) * D - Sa_b) ** 2) +
               np.sqrt(hu_b ** 2 + (l3_b - index_m * D + A * np.cos(beta_bed)) ** 2) -
               np.sqrt(hb_b ** 2 + (l3_b - index_m * D) ** 2) -
               np.sqrt(hu_b ** 2 + (l3_b - (index_m - 1) * D - Sa_b + A * np.cos(beta_bed)) ** 2)) / (2 * A)
    else:
        F_m = (np.sqrt(hb ** 2 + (l3_b - (index_m - 1) * D - Sa_b) ** 2) +
               np.sqrt(hu ** 2 + (l3_b - index_m * D + A * np.cos(beta)) ** 2) -
               np.sqrt(hb ** 2 + (l3_b - index_m * D) ** 2) -
               np.sqrt(hu ** 2 + (l3_b - (index_m - 1) * D - Sa_b + A * np.cos(beta)) ** 2)) / (2 * A)
    F_m[~mask] = 0
    F_pv_front_m_nsgnd = F_m.sum(axis=1)

    # VF of the PV array to ground
    F_pv_front_gnd = np.where(n == 1, (1 - np.cos(beta)) / 2,
                              (A + np.sqrt(lf ** 2 + hb ** 2) -
                               np.sqrt((lf + A * np.cos(beta)) ** 2 + hu ** 2)) / (2 * A))

    # VF of the PV array to the illuminated ground
    F_pv_front_nsgnd = np.where(np.logical_or(np.logical_and(Sa > 0, le >= (n - 1) * D - Sb),
                                              np.logical_and(Sa < 0, lf <= -((N - n) * D + np.abs(Sa) + Sb))),
                                F_pv_front_gnd,
                                F_pv_front_f_nsgnd + F_pv_front_m_nsgnd + F_pv_front_r_nsgnd)

    return F_pv_front_nsgnd


def cal_F_PV_nsgnd_rear(n, N, D, A, h, beta, azimuth_PV, solar_azimuth, solar_altitude):
    hu, hb, lf, lr, le = cal_relative_parameter(D, A, h, beta)
    Sa = A * np.cos(beta) + A * np.sin(beta) * np.cos(solar_azimuth - azimuth_PV) / np.tan(solar_altitude)
    Sb = np.where(Sa > 0, hb * np.cos(solar_azimuth - azimuth_PV) / np.tan(solar_altitude),
                  A * np.cos(beta) + hu * np.cos(solar_azimuth - azimuth_PV) / np.tan(solar_altitude))

    # rear head VF
    lr = np.where(n == N, float('inf'), lr)
    lp = np.ceil((le + Sb) / D) * D - (le + Sb)
    l1 = np.minimum(le + lp - np.abs(Sa), le)
    F_pv_rear_f_nsgnd = np.where(le > (n - 1) * D - Sb,
                                 A + np.sqrt(hb ** 2 + ((n - 1) * D - Sb) ** 2) -
                                 np.sqrt(hu ** 2 + ((n - 1) * D - Sb + A * np.cos(beta)) ** 2),
                                 np.where(np.logical_and(le > -((N - n) * D + Sb), np.abs(Sa) < D),
                                          np.sqrt(hb ** 2 + (le + lp - D) ** 2) +
                                          np.sqrt(hu ** 2 + (l1 + A * np.cos(beta)) ** 2) -
                                          np.sqrt(hb ** 2 + l1 ** 2) -
                                          np.sqrt(hu ** 2 + (le + lp - D + A * np.cos(beta)) ** 2), 0)) / (2 * A)
    # rear tail VF
    lq = lr - Sb - np.floor((lr - Sb) / D) * D
    l2 = np.maximum(lr - lq + np.abs(Sa), lr)
    F_pv_rear_r_nsgnd = np.where(lr >= (N - n) * D + np.abs(Sa) + Sb,
                                 np.where(n == N,
                                          np.sqrt(hu ** 2 + (np.abs(Sa) + Sb - A * np.cos(beta)) ** 2) -
                                          np.sqrt(hb ** 2 + (np.abs(Sa) + Sb) ** 2) + A * np.cos(beta),
                                          np.sqrt(hb ** 2 + lr ** 2) +
                                          np.sqrt(hu ** 2 + ((N - n) * D + np.abs(Sa) + Sb - A * np.cos(beta)) ** 2) -
                                          np.sqrt(hb ** 2 + ((N - n) * D + np.abs(Sa) + Sb) ** 2) -
                                          np.sqrt(hu ** 2 + (lr - A * np.cos(beta)) ** 2)),
                                          np.where(np.abs(Sa) < D,
                                                   np.sqrt(hb ** 2 + l2 ** 2) +
                                                   np.sqrt(hu ** 2 + (lr - lq + np.abs(Sa) - A * np.cos(beta)) ** 2) -
                                                   np.sqrt(hb ** 2 + (lr - lq + np.abs(Sa)) ** 2) -
                                                   np.sqrt(hu ** 2 + (l2 - A * np.cos(beta)) ** 2), 0)) / (2 * A)
    #  rear middle VF
    K = np.minimum(np.floor((le + Sb) / D), n - 1) + np.minimum(np.floor((lr - Sb) / D), N - n)
    maxK = K.max()
    index_k = np.arange(1, maxK + 1)
    l3_b = np.where(le < (n - 1) * D - Sb, le + lp - D, (n - 1) * D - Sb)[:, None]
    Sa_b = np.abs(Sa).values[:, None]
    valid = ((K >= 1) & (np.abs(Sa) < D)).values
    mask = (index_k[None, :] <= K[:, None]) & valid[:, None]  # shape (N, maxM)

    if isinstance(beta, np.ndarray):
        beta_b = beta[:, None]
        hb_b = hb[:, None]
        hu_b = hu[:, None]
        F_k = (np.sqrt(hb_b**2 + (l3_b - index_k * D)**2) +
               np.sqrt(hu_b**2 + (l3_b + A * np.cos(beta_b) - (index_k - 1)*D - Sa_b)**2) -
               np.sqrt(hb_b**2 + (l3_b - (index_k - 1)*D - Sa_b)**2) -
               np.sqrt(hu_b**2 + (l3_b + A * np.cos(beta_b) - index_k * D)**2)) / (2 * A)
    else:
        F_k = (np.sqrt(hb ** 2 + (l3_b - index_k * D) ** 2) +
               np.sqrt(hu ** 2 + (l3_b + A * np.cos(beta) - (index_k - 1) * D - Sa_b) ** 2) -
               np.sqrt(hb ** 2 + (l3_b - (index_k - 1) * D - Sa_b) ** 2) -
               np.sqrt(hu ** 2 + (l3_b + A * np.cos(beta) - index_k * D) ** 2)) / (2 * A)

    F_k[~mask] = 0
    F_pv_rear_m_nsgnd = F_k.sum(axis=1)

    # VF of the PV array to ground
    F_pv_rear_gnd = np.where(n < N,
                             (A + np.sqrt(hb ** 2 + lr ** 2) -
                              np.sqrt(hu ** 2 + (lr - A * np.cos(beta)) ** 2)) / (2 * A),
                             (1 + np.cos(beta)) / 2)
    # VF of the PV array to the illuminated ground
    F_pv_rear_nsgnd = np.where(np.logical_or(np.logical_and(Sa > 0, lr <= -((n - 1) * D - Sb)),
                                             np.logical_and(Sa < 0, le <= -(np.abs(Sa) + Sb + (N - n) * D))),
                               F_pv_rear_gnd,
                               F_pv_rear_f_nsgnd + F_pv_rear_m_nsgnd + F_pv_rear_r_nsgnd)

    return F_pv_rear_nsgnd

