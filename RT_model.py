# The RT simulations were performed using bifacial_radiance v0.4.4 with RADIANCE 6.0a (2025-06-08; 6.0.78fec72ee8)

import bifacial_radiance as br
import os
from math import degrees


import bifacial_radiance as br
import os
from math import degrees


def cal_irradiance(n, N, D, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth,
                   H_length, V_length, V_num,
                   xgap=None, ygap=None, zgap=None, N_mod=None,
                   project_name=None, save_path=None, accuracy='low'):
    """
    :param V_length: module vertical length
    :param H_length: module horizontal length
    :param V_num: The number of rows in the vertical direction of the array
    :param xgap: horizontal distance between modules
    :param ygap: vertical distance between modules
    :param zgap: Distance between module and torque tube
    :param n: Target PV array location
    :param N: Total number of rows of PV field
    :param D: row spacing
    :param h: hub height
    :param beta: tilt
    :param azimuth: azimuth
    :param N_mod: The number of components in a row
    :param Albedo: albedo
    :param DNI: direct normal irradiance
    :param DHI: horizontal diffuse irradiance
    :param solar_altitude: solar altitude
    :param solar_azimuth: solar azimuth
    :param project_name: project name
    :param save_path: save path
    :param accuracy: Simulation accuracy
    :return:
    """

    solar_altitude = degrees(solar_altitude)
    solar_azimuth = degrees(solar_azimuth)
    beta = degrees(beta)
    azimuth = degrees(azimuth) + 180

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('----------Folder created--------------')
    # 1️⃣ 初始化工程
    demo = br.RadianceObj(project_name, save_path)
    demo.setGround(Albedo)

    # 生成天空
    demo.gendaylit2manual(DNI, DHI, solar_altitude, solar_azimuth)

    # create a scene using panels in landscape at 10 deg tilt, 1.5m pitch. 0.2 m ground clearance
    moduletype = 'test-module'

    module = demo.makeModule(name=moduletype, x=H_length, y=V_length,
                             numpanels=V_num, xgap=xgap, ygap=ygap, zgap=zgap)

    sceneDict = {'tilt': beta, 'pitch': D, 'hub_height': h, 'azimuth': azimuth, 'nMods': N_mod, 'nRows': N}

    # makeScene creates a .rad file with 10 modules per row, 3 rows.
    scene = demo.makeScene(module=module, sceneDict=sceneDict)

    # makeOct combines all of the ground, sky and object files into .oct file.
    octfile = demo.makeOct(demo.getfilelist())

    # return an analysis object including the scan dimensions for back irradiance
    analysis = br.AnalysisObj(octfile, demo.name)

    frontscan, backscan = analysis.moduleAnalysis(scene, sensorsy=9, rowWanted=n)

    analysis.analysis(octfile, demo.name, frontscan, backscan, accuracy=f'{accuracy}')

    front_irr = analysis.Wm2Front
    rear_irr = analysis.Wm2Back

    return front_irr, rear_irr


if __name__ == "__main__":
    # 测试程序
    (V_length, H_length, V_num, xgap, ygap, zgap) = [1.674, 1.002, 2, 0.02, 0.145, 0.115]
    (n, N, D, h, beta, azimuth, N_mod) = [4, 8, 7.6, 2.3, 25, 0, 11]
    (Albedo, DNI, DHI, solar_altitude, solar_azimuth) = [0.206448, 33.1, 168.2, 40.50, 50]
    (project_name, save_path, accuracy) = ['PV_test', 'test', 'high']

    front_irr, rear_irr = cal_irradiance(n, N, D, h, beta, azimuth, Albedo, DNI, DHI, solar_altitude, solar_azimuth,
                       H_length, V_length, V_num,
                       xgap, ygap, zgap, N_mod,
                       project_name, save_path, accuracy='low')
  
