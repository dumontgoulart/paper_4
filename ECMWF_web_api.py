# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:04:50 2020

@author: tbr910
"""
import sys

import os
import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import seaborn as sns
import pandas as pd


from ecmwfapi import ECMWFService
from ecmwfapi import ECMWFDataServer
#import xarray as xr 

###################################################################### MARS ARCHIVE ##########################################################################
os.chdir(r'D:\paper_4\data\seas5\ecmwf') ## Dir has to be tbr910, for some weird reason
################################# SEAS forecast ############################################### 

server = ECMWFService("mars")
lon_min, lat_min, lon_max, lat_max = -88.549805,20.303418,-58.447266,31.391158
start_date = '2017-09-06'
step_ini, step_end, step_ratio = 0, 24, 1
number = 1
grid = 0.25
vars = 'tp'

dict_vars = {'tp': '228.128', 'msl': '151.128', 'u10': '165.128', 'v10': '166.128', 't2m': '167.128'}

server.execute(
    {
    "class": "od",
    "date": f"{datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')}",
    "expver": "1",
    "levtype": "sfc",
    "number": f"1/to/{number}",
    "param": f"{dict_vars[vars]}",
    "step": f"{step_ini}/to/{step_end}/by/{step_ratio}",
    "stream": "enfo",
    "time": "00",
    "type": "pf",
    "grid": f"{grid}/{grid}",
    "area": f"{int(lat_min)}/{int(lon_min)}/{int(lat_max)}/{int(lon_max)}",
    "format": "netcdf",
    },
    f"ecmwf_eps_pf_{vars}_n{number}_s{step_end}_{datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')}.nc")







