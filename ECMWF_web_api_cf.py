# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:04:50 2020

@author: tbr910
"""
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd


from ecmwfapi import ECMWFService
from ecmwfapi import ECMWFDataServer

###################################################################### MARS ARCHIVE ##########################################################################
os.chdir(r'/scistor/ivm/hmt250/ecmwf/data') ## Dir has to be tbr910, for some weird reason
################################# SEAS forecast ############################################### 

server = ECMWFService("mars")


# Now with ensemble forecast

dict_storm = {'sandy':{'storm_date_start':'2012-10-22','storm_date_end':'2012-10-30', 'plot_day':'2012-10-25', 'sid':'2012296N14283', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'}, 
              'igor':{'storm_date_start':'2010-09-11','storm_date_end':'2010-09-22', 'plot_day':'2010-09-19', 'sid':'2010251N14337', 'bbox':[-89.472656,7.532249,-30.078125,55], 'target_city':'New York'},
              'earl':{'storm_date_start':'2010-08-29','storm_date_end':'2010-09-04', 'plot_day':'2010-09-04', 'sid':'2010236N12341', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'irene':{'storm_date_start':'2011-08-22','storm_date_end':'2011-08-28', 'plot_day':'2011-08-27', 'sid':'2011233N15301', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'katia':{'storm_date_start':'2011-09-02','storm_date_end':'2011-09-10', 'plot_day':'2011-09-06', 'sid':'2011240N10341', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'ophelia':{'storm_date_start':'2011-09-28','storm_date_end':'2011-10-03', 'plot_day':'2011-10-02', 'sid':'2011263N12323', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'rafael':{'storm_date_start':'2012-10-12','storm_date_end':'2012-10-24', 'plot_day':'2012-10-16', 'sid':'2012287N15297', 'bbox':[-85.473633,7.532249,-30.078125,56], 'target_city':'Santiago de Cuba'},
              'gonzalo':{'storm_date_start':'2014-10-13','storm_date_end':'2014-10-18', 'plot_day':'2014-10-16', 'sid':'2014285N16305', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'Santiago de Cuba'},
              'hayan':{'storm_date_start':'2013-11-04','storm_date_end':'2013-11-09','plot_day':'2013-11-07', 'sid':'2013306N07162', 'bbox':[117.641602,6.271618,131.308594,19.932041],'target_city':'Tacloban'}, 
              'xynthia':{'storm_date_start':'2010-02-27','storm_date_end':'2010-03-01','plot_day':'2010-02-27', 'sid':'xynthia_track', 'bbox':[-45.771484,22.431340,29.707031,64.227957],'target_city':'La Rochelle'},
              'xaver':{'storm_date_start':'2013-12-04','storm_date_end':'2013-12-07','plot_day':'2013-12-05', 'sid':'xaver_track', 'bbox':[-46.982422,22.979488,36.826172,69.886265],'target_city':'Hamburg'}, 
              'megi':{'storm_date_start':'2010-10-12','storm_date_end':'2010-10-24','plot_day':'2010-10-20', 'sid':'2010285N13145', 'bbox':[117.597656,6.664608,133.154297,18.729502],'target_city':'Tacloban'}, 
              'yasi':{'storm_date_start':'2011-01-31','storm_date_end':'2011-02-05','plot_day':'2011-02-02', 'sid':'2011028S13180', 'bbox':[144.316406,-21.779905,161.718750,-10.790141],'target_city':'Townsville'}, 
              'idai':{'storm_date_start':'2019-03-13','storm_date_end':'2019-03-16','plot_day':'2019-03-14','sid':'2019063S17066','bbox':[31.673584,-22.309426,45.483398,-14.392118],'target_city':'Beira'}} 


storm = 'idai'

lon_min, lat_min, lon_max, lat_max = dict_storm[storm]['bbox'][0],dict_storm[storm]['bbox'][1],dict_storm[storm]['bbox'][2],dict_storm[storm]['bbox'][3]

start_dates = ['2019-03-13']#, '2019-03-14', '2019-03-15', '2019-03-16']
times = ["00:00:00"]#, "06:00:00", "12:00:00", "18:00:00"]
step_ini, step_end, step_ratio = 0, 90, 1
grid = 0.1
vars = ['tp', 'u10', 'v10', 'msl', 'tpr'] 
dict_vars = {'tp': '228.128', 'msl': '151.128', 'u10': '165.128', 'v10': '166.128', 't2m': '167.128', 'tpr': '260048'}


for start_date in start_dates:
    for time in times:
        # Construct the "param" value with variables separated by "/"
        param_value = "/".join(dict_vars[var] for var in vars)

        output_filename = f"ecmwf_eps_cf_010_vars_s{step_end}_{datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')}_{times[0][0:2]}.nc"

        server.execute(
            {
                "class": "od",
                "date": f"{datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')}",
                "expver": "1",
                "levtype": "sfc",
                "param": param_value,
                "step": f"{step_ini}/to/{step_end}/by/{step_ratio}",
                "stream": "enfo",
                "time": f"{time}",
                "type": "cf",
                "grid": f"{grid}/{grid}",
                "area": f"{int(lat_min)}/{int(lon_min)}/{int(lat_max)}/{int(lon_max)}",
                "format": "netcdf",
            },
            output_filename
        )