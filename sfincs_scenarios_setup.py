# -*- coding: utf-8 -*-
"""
Prepare files for HydroMT
Created on Tue Oct  9 16:39:47 2022
@author: morenodu
"""
import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import configparser
import yaml

import hydromt
from hydromt_sfincs import SfincsModel, utils

import sfincs_scenario_functions as sfincs_scen


##########################################################################################
# GENERATE HYDROMT CONFIG FILE
fn_exe = r"D:\paper_4\data\sfincs_model\SFINCS_v2.0.3_Cauberg_release_exe\sfincs.exe"
root_folder = 'D:/paper_4/data/sfincs_input/'  #'D:/paper_4/data/version006_quadtree_8km_500m_60s/'
output_folder = rf'{root_folder}ini_test'
os.chdir(root_folder)

# general parameters
res = 30
storm = 'idai'
bbox_beira = [34.8180412,-19.8658097,34.939334,-19.76172]
grid_setup = {
    "x0": 794069.0,
    "y0": 7416897.0,
    "dx": 8000.0,
    "dy": 8000.0,
    "nmax": 77,
    "mmax": 112,
    "rotation": 55,
    "epsg": 32736,}

# offshore simulation dates
tref_off = '20190301 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
tstart_off = '20190313 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
tstop_off = '20190317 000000' # Keep 20190317 120000 for offshore model - it doesn't work with other dates
hours_shifted = -137 # hour for peak tide

# onshore simulation dates
tref = '20190314 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
tstart = '20190314 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
tstop = '20190316 000000' # Keep 20190317 120000 for offshore model - it doesn't work with other dates

# libraries
data_libs = ['d:/paper_4/data/data_catalogs/data_catalog_converter.yml', root_folder+f'data_deltares_{storm}/data_catalog.yml']
list_indices_storm = ['merit_hydro','gebco', 'osm_coastlines', 'osm_landareas', 'gswo', 'fabdem', 'dtu10mdt', 'gcn250', 'vito', "rivers_lin2019_v1"]

##########################################################################################
# optional step: create data catalog
sfincs_scen.clip_data_to_region(bbox_beira, export_dir = f'data_deltares_test', data = ['deltares_data', data_libs[0]], list_indices = list_indices_storm)

# 1) Offshore model
sfincs_scen.create_sfincs_base_model(root_folder = root_folder, scenario = 'era5', storm = storm, data_libs = data_libs,
                         bbox = None, grid_setup = grid_setup, topo_map = 'beira_dem', res = res,
                         tref = tref_off, tstart = tstart_off, tstop = tstop_off)


sfincs_scen.generate_forcing(grid_setup, root_folder,'base', r'D:\paper_4\data\quadtree_ifs_cf_forcing_bc',
                 data_libs, tref_off, tstart_off, tstop_off, forcing_catalog = 'ifs_cf_ens_idai_hourly',
                 forcings=['wind', 'pressure'])

# copy the folders and external forcings
sfincs_scen.copy_entire_folder(rf'D:\paper_4\data\quadtree_beira_base', rf'D:\paper_4\data\quadtree_ifs_cf_forcing_bc')
sfincs_scen.copy_two_files(rf'D:\paper_4\data\quadtree_ifs_cf_forcing_bc', rf'D:\paper_4\data\quadtree_ifs_cf_bc', 'press_2d.nc', 'wind_2d.nc')

# run model and generate his (waterlevels)
sfincs_scen.run_sfincs(base_root = r'D:\paper_4\data\quadtree_era5_max_tide', fn_exe = fn_exe)

# 2) Onshore model
sfincs_scen.create_sfincs_base_model(root_folder = root_folder, scenario = 'base', storm = storm, data_libs = data_libs,
                         bbox = bbox_beira, topo_map = 'beira_dem', res = res,
                         tref = tref, tstart = tstart, tstop = tstop)

sfincs_scen.update_sfincs_model(base_root = f'{root_folder}{storm}_base', new_root = f'{root_folder}{storm}_surge_ifs_cf_bc', 
                    data_libs = data_libs, mode = 'surge', waterlevel_path = r"D:/paper_4/data/quadtree_ifs_cf_bc/sfincs_his.nc")


# run onshore model and obtain inundation
sfincs_scen.run_sfincs(base_root = r'D:\paper_4\data\sfincs_input\test_surge_ifs_cf_bc_floodwall', fn_exe = fn_exe) # test_slr100_surge # test_rain_gpm
