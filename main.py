"""
Author Zexing Deng
time 2025-11-17
This file used to calculate the irradiance of the PV array
"""

import irradiance_simulate as irs
import pandas as pd
import numpy as np

# F25
beta = 25
azimuth = 0
D = 7.6
A = 3.04
h = 2.3
n = 4
N = 8

Dm_data = pd.read_csv('data/example.csv', index_col=0)
Dm_data.index = pd.to_datetime(Dm_data.index)

DNI = Dm_data['DNI']
DHI = Dm_data['DHI']
Albedo = Dm_data['albedo']
solar_altitude = Dm_data['sunalt']
solar_azimuth = Dm_data['sunazi']
mea_front = Dm_data['GpoaF']
mea_rear = Dm_data['GpoaR']

beta, azimuth = irs.cal_array_angle(0, beta, azimuth)

sim_f = irs.cal_irr_front(n, N, D, A, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth)
sim_f = pd.Series(np.sum(sim_f, axis=0), index=mea_front.index)
sim_r = irs.cal_irr_rear(n, N, D, A, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth)
sim_r = pd.Series(np.sum(sim_r, axis=0), index=mea_front.index)

data = data = pd.DataFrame({'mea_front': mea_front,
                            'sim_front': sim_f,
                            'mea_rear': mea_rear,
                            'sim_rear': sim_r})
data.to_csv('data/Dm_fixed_simulated.csv')

