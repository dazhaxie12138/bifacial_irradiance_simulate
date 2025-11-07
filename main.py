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

Dm_data = pd.read_csv('data/Dm_fixed_cleaned.csv', index_col=0)
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


# HSATs
'''
beta_max = 60
azimuth = 0
D = 12
A = 3.3
h = 1.95
n = 3
N = 8

Dm_data = pd.read_csv('data/Dm_hast_cleaned.csv', index_col=0)
Dm_data.index = pd.to_datetime(Dm_data.index)

DNI = Dm_data['DNI']
DHI = Dm_data['DHI']
Albedo = Dm_data['albedo']
solar_altitude = Dm_data['sunalt']
solar_azimuth = Dm_data['sunazi']
mea_front = Dm_data['GpoaF']
mea_rear = Dm_data['GpoaR']

beta, azimuth = irs.cal_array_angle(2, beta_max, azimuth, D, A, solar_altitude, solar_azimuth)

sim_f = irs.cal_irr_front(n, N, D, A, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth)
sim_f = pd.Series(np.sum(sim_f, axis=0), index=mea_front.index)
sim_r = irs.cal_irr_rear(n, N, D, A, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth)
sim_r = pd.Series(np.sum(sim_r, axis=0), index=mea_front.index)
'''

# NREL-H
'''
beta = 89.99
azimuth = 3
mode = 0
D = 3.05
A = 0.61
h = 0.61*1.5
n = 2
N = 3
H_length, V_length, V_num, xgap, ygap, zgap, N_mod = (0.61, 0.61, 1, 0.01, 0, 0.1, 10)
project_name, accuracy = 'NREL_facing_NS',  'low'
NREL_data = pd.read_csv('data/NREL_facing_south_data.csv', index_col=0)
NREL_data.index = pd.to_datetime(NREL_data.index)

DNI = NREL_data['DNI']
DHI = NREL_data['DHI']
Albedo = NREL_data['albedo']
solar_altitude = NREL_data['sunalt']
solar_azimuth = NREL_data['sunazi']
mea_front = NREL_data['GpoaF']
mea_rear = NREL_data['GpoaR']

beta, azimuth = irs.cal_array_angle(0, beta, azimuth, D, A, solar_altitude, solar_azimuth)

sim_f = irs.cal_irr_front(n, N, D, A, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth)
sim_f = pd.Series(np.sum(sim_f, axis=0), index=mea_front.index)
sim_r = irs.cal_irr_rear(n, N, D, A, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth)
sim_r = pd.Series(np.sum(sim_r, axis=0), index=mea_front.index)
'''


