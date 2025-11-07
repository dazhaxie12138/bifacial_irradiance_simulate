"""
from the article
Detailed mathematical simulation model of incident irradiance
on photovoltaic arrays considering finite number of rows and array locations
"""

import numpy as np
from math import sin, radians, pi
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
import view_factor as vf
import Perez_continuous as Perez


def cal_array_angle(mode, beta, azimuth, D=None, A=None, solar_altitude=None, solar_azimuth=None):
    """
    :param mode: operation mode,0-fixed tilt；1-horizontal single-axis true tracking；2-horizontal single-axis backtracking
    :param D: row spacing
    :param A: height of PV array
    :param solar_altitude:
    :param solar_azimuth:
    :param beta: tilt of PV array when mode == 0
    :return: tilt and azimuth of PV array
    """
    beta = radians(beta)
    azimuth = radians(azimuth)
    if mode == 0:
        return beta, azimuth
    else:
        # cited from "Performance Analysis of a Double-Sided PV Plant Oriented with Backtracking System"
        normal_tracking_mode = np.abs(np.arctan(np.sin(solar_azimuth) / np.tan(solar_altitude)))
        normal_tracking_mode = np.minimum(normal_tracking_mode, beta)
        normal_tracking_mode[solar_altitude < 0] = beta

        # cited from "Optimal design and cost analysis of single-axis tracking photovoltaic power plants"
        Sa = (A * np.cos(normal_tracking_mode) +
              A * np.sin(normal_tracking_mode) / np.tan(solar_altitude) * np.sin(solar_azimuth))
        backtracking_mode = np.abs(pi / 2 - solar_altitude - np.arccos(D / A * np.sin(solar_altitude)))
        backtracking_mode = np.where(Sa > D, backtracking_mode,
                            np.where(np.logical_and(Sa <= D, backtracking_mode < normal_tracking_mode),
                                     backtracking_mode, normal_tracking_mode))  # 回溯跟踪
        # calculated the azimuth
        azimuth = pd.Series(np.where(solar_azimuth < 0, -pi / 2, pi / 2), index=solar_altitude.index)
        backtracking_mode = pd.Series(backtracking_mode, index=solar_altitude.index)
        normal_tracking_mode = pd.Series(normal_tracking_mode, index=solar_altitude.index)
        if mode == 1:
            return normal_tracking_mode, azimuth
        else:
            return backtracking_mode, azimuth


def cal_irr_front(n, N, D, A, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth):
    '''
    :param n:光伏阵列第几行
    :param N:光伏方阵的总行数
    :param D:行间距
    :param A:光伏阵列宽度
    :param h:光伏阵列安装高度
    :param beta:倾角
    :param azimuth:方位角
    :param Albedo:地面反照率
    :param DNI:直接法向辐照
    :param DHI:水平漫射辐照
    :param F1:周太阳系数
    :return:正面直接辐照、正面周太阳辐照、正面各向同性漫射辐照、正面反射辐照
    '''
    if isinstance(beta, pd.Series):
        n = np.where(solar_azimuth < 0, n, N - n + 1)
        beta = beta.values
        azimuth = azimuth.values
    cos_sita = (np.sin(solar_altitude) * np.cos(beta) +
                np.cos(solar_altitude) * np.sin(beta) * np.cos(solar_azimuth - azimuth))
    # direct irradiance
    k1 = np.maximum(cos_sita, 0)
    front_dir = k1 * DNI

    # diffuse irradiance
    F1 = Perez.PerezDriesseContinuous(DNI, DHI, solar_altitude)[0]
    F_pv_front_sky = np.where(n == 1, (1 + np.cos(beta)) / 2,
                              (A + D - np.sqrt((A * np.sin(beta)) ** 2 + (D - A * np.cos(beta)) ** 2)) / (2 * A))
    front_dif_BEA = np.array(F1 * k1 / np.maximum(sin(radians(5)), np.sin(solar_altitude)) * DHI)
    front_dif_ISO = np.array((1 - F1) * F_pv_front_sky * DHI)

    # beam-reflected irradiance
    Beam_ground = (DNI + F1 * DHI / np.maximum(sin(radians(5)), np.sin(solar_altitude))) * np.sin(solar_altitude)
    F_pv_front_nsgnd = vf.cal_F_PV_nsgnd_front(n, N, D, A, h, beta, azimuth, solar_azimuth, solar_altitude)
    front_ref_BEA = Albedo * F_pv_front_nsgnd * Beam_ground

    #  diffuse-reflected irradiance
    F_pv_front_dx_sky = vf.cal_F_pv_front_dx_sky(n, N, D, A, h, beta, solar_altitude)
    front_ref_ISO = Albedo * F_pv_front_dx_sky * (1 - F1) * DHI

    # reflected irradiance
    front_ref = np.array(front_ref_ISO + front_ref_BEA)
    return front_dir, front_dif_BEA, front_dif_ISO, front_ref


def cal_irr_rear(n, N, D, A, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth):

    if isinstance(beta, pd.Series):
        n = np.where(solar_azimuth < 0, n, N - n + 1)
        beta = beta.values
        azimuth = azimuth.values

    # direct irradiance
    cos_sita = (np.sin(solar_altitude) * np.cos(beta + pi) +
                np.cos(solar_altitude) * np.sin(beta + pi) * np.cos(solar_azimuth - azimuth))
    k1 = np.maximum(cos_sita, 0)
    rear_dir = k1 * DNI

    # diffuse irradiance
    F1 = Perez.PerezDriesseContinuous(DNI, DHI, solar_altitude)[0]
    F_pv_rear_sky = np.where(n == N, (1 - np.cos(beta)) / 2,
                             (A + D - np.sqrt((D + A * np.cos(beta)) ** 2 + (A * np.sin(beta)) ** 2)) / (2 * A))

    rear_dif_BEA = np.array(F1 * k1 / np.maximum(sin(radians(5)), np.sin(solar_altitude)) * DHI)
    rear_dif_ISO = np.array((1 - F1) * F_pv_rear_sky * DHI)

    # beam-reflected irradiance
    Beam_ground = (DNI + F1 * DHI / np.maximum(sin(radians(5)), np.sin(solar_altitude))) * np.sin(solar_altitude)
    F_pv_rear_nsgnd = vf.cal_F_PV_nsgnd_rear(n, N, D, A, h, beta, azimuth, solar_azimuth, solar_altitude)
    rear_ref_BEA = Albedo * F_pv_rear_nsgnd * Beam_ground

    # diffuse-reflected irradiance
    F_pv_rear_dx_sky = vf.cal_F_pv_rear_dx_sky(n, N, D, A, h, beta, solar_altitude)
    rear_ref_ISO = Albedo * F_pv_rear_dx_sky * (1 - F1) * DHI

    # reflected irradiance
    rear_ref = np.array(rear_ref_BEA + rear_ref_ISO)

    return rear_dir, rear_dif_BEA, rear_dif_ISO, rear_ref

