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

sys.path.insert(0, 'D:/paper_3/code')
import clip_deltares_data_to_region as clip
import hydromt_sfincs_pipeline as sfincs_pipe

def generate_forcing(grid_setup, root_folder, scenario, new_root, data_libs, tref, tstart, tstop, forcings=['wind', 'precip', 'pressure']):
    root=f'{root_folder}offshore_forcings_{scenario}'
    sf = SfincsModel(data_libs = data_libs, root = root, mode="w+")
    # Set up the grid
    sf.setup_grid(**grid_setup)
    # Add time-series:
    start_datetime = datetime.strptime(tstart, '%Y%m%d %H%M%S')
    stop_datetime = datetime.strptime(tstop, '%Y%m%d %H%M%S')
    sf.setup_config(
        **{
            "tref": tref,
            "tstart": tstart,
            "tstop": tstop,
            "dtout":10800,
            "dtmaxout": (stop_datetime - start_datetime).total_seconds(),
            "advection": 0,
        }
    )
    # ERA5
    if 'precip' in forcings:
        sf.setup_precip_forcing_from_grid(precip=f'era5_hourly_test', aggregate=False)
    if 'wind' in forcings:
        sf.setup_wind_forcing_from_grid(wind=f'era5_hourly_test')
    if 'pressure' in forcings:
        sf.setup_pressure_forcing_from_grid(press = f'era5_hourly_test')

    sf.set_root(root = new_root, mode='w+')    

    # save model
    sf.write()  # write all because all folders are run in parallel on snellius

def shift_time_and_save(input_file, output_file, days):
    # Open the netCDF file
    ds = xr.open_dataset(input_file)
    with xr.open_dataset(input_file) as ds:
        data = ds.load()
    ds.close()

    # Shift the time dimension by specified number of days
    ds_shifted = data.assign_coords(time=data.time - pd.Timedelta(days=days))

    # Save the shifted Dataset to a new netCDF file
    ds_shifted.to_netcdf(output_file)
    ds_shifted.close()

def shift_forcings(folder, vars = ['precip', 'press', 'wind'], days = 5):
    for var in vars:
        shift_time_and_save(f'{folder}/{var}_2d.nc', f'{folder}/{var}_2d.nc', days = days)

##########################################################################################
# GENERATE HYDROMT CONFIG FILE
fn_exe = r"D:\paper_4\data\sfincs_model\SFINCS_v2.0.3_Cauberg_release_exe\sfincs.exe"
root_folder = 'D:/paper_4/data/sfincs_input/'  #'D:/paper_4/data/version006_quadtree_8km_500m_60s/'
output_folder = rf'{root_folder}ini_test'
os.chdir(root_folder)

# sfincs model -19.7824710,34.8180412
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

tref_off = '20190301 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
tstart_off = '20190308 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
tstop_off = '20190317 000000' # Keep 20190317 120000 for offshore model - it doesn't work with other dates

tref = '20190313 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
tstart = '20190313 000000' # Keep 20190301 000000 for offshore model - it doesn't work with other dates
tstop = '20190316 000000' # Keep 20190317 120000 for offshore model - it doesn't work with other dates
res = 30
storm = 'test'
data_libs = ['d:/paper_4/data/data_catalogs/data_catalog_converter.yml', root_folder+f'data_deltares_{storm}/data_catalog.yml']
list_indices_storm = ['merit_hydro','gebco', 'osm_coastlines', 'osm_landareas', 'gswo', 'fabdem', 'dtu10mdt', 'gcn250', 'vito', "rivers_lin2019_v1"]

# #### this takes a lot of time so do it once
# clip.clip_data_to_region(bbox_beira, export_dir = f'data_deltares_test', data = ['deltares_data', data_libs[0]], list_indices = list_indices_storm)
# clip.correct_fabdem_nas(path_folder = f'data_deltares_test', path_catalogue = f'data_deltares_test/data_catalog.yml')
# ####
##########################################################################################


# 2) GENERATE offshore SFINCS MODEL base
sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = 'base', storm = storm, data_libs = data_libs, 
                                  area_mask = None,bbox = None, grid_setup = grid_setup, writing_mode = 'write', 
                                  topo_map = 'beira_dem', res = res, tref = tref_off, tstart = tstart_off, tstop = tstop_off)

# # get the forcing from ERA5
clip.data_catalog_edit(rf'd:\paper_4\data\data_catalogs\data_catalog_converter.yml', 'era5_hourly_sample', 
                       f'era5_hourly_2x_test', f'd:/paper_4/data/era5/era5_hourly_vars_idai_single_2019_03.nc')

# create the forcings for generating the waterlevels based on era5 data
sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = 'era5', storm = storm, data_libs = data_libs,
                                  area_mask = None,bbox = None, grid_setup = grid_setup, writing_mode = 'update', topo_map = 'beira_dem', 
                                  res = res, mode = 'era5', tref = tref_off, tstart = tstart_off, tstop = tstop_off)


# ####### Create forcings for offshore model matching the grid
# generate_forcing(grid_setup, root_folder, scenario = 'base', new_root=r'D:\paper_4\data\quadtree_era5', data_libs, tref, tstart, tstop, forcings=['wind', 'precip', 'pressure'])
# max tide scenario: change the external wind and pressure values from the 14th to the 9th, so in terms of total days: 5 days
#TODO: not working; the inp file is being generated but we want the old inp file + the added forcings, so change that.
generate_forcing(grid_setup, root_folder, 'max_tide', rf'D:\paper_4\data\quadtree_max_tide', data_libs, tref_off, tstart_off, tstop_off, forcings=['wind', 'precip', 'pressure'])

# shift the forcings 5 days
# shift_forcings(folder = rf'D:\paper_4\data\quadtree_max_tide', vars = ['precip', 'press', 'wind'], days = 5)
# #######

# # 2) RUN OFFSHORE SFINCS MODEL 
#TODO: this should not be a series of running but a single run based on different scenario options
# sfincs_pipe.run_sfincs(base_root = 'D:\paper_4\data\quadtree_era5', fn_exe = fn_exe)
# Run offshore sfincs model with max tide
# sfincs_pipe.run_sfincs(base_root = r'D:\paper_4\data\quadtree_max_tide', fn_exe = fn_exe)

# 3a) generate base SFINCS model for onshore
sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = 'base', storm = storm, data_libs = data_libs, 
                                  area_mask = None,bbox = bbox_beira, grid_setup = None, writing_mode = 'write', 
                                  topo_map = 'beira_dem', res = res, tref = tref, tstart = tstart, tstop = tstop)

# 3b) add surge forcing to ONSHORE SFINCS models
sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = 'base', storm = storm, data_libs = data_libs,
                                  area_mask = None,bbox = bbox_beira, grid_setup = None, writing_mode = 'update', 
                                  topo_map = 'beira_dem', res = res, mode = 'sfincs_surge', tref = tref, tstart = tstart, tstop = tstop,
                                  offshore_path = r"D:/paper_4/data/quadtree_era5/sfincs_his_spider.nc")
# now try with slr = 1
sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = 'slr100', storm = storm, data_libs = data_libs,
                                    area_mask = None,bbox = bbox_beira, grid_setup = None, writing_mode = 'update',
                                    topo_map = 'beira_dem', res = res, mode = 'sfincs_surge', tref = tref, tstart = tstart, tstop = tstop,
                                    slr = 1, offshore_path = r"D:/paper_4/data/quadtree_era5/sfincs_his_spider.nc")
# now with RAIN ERA5
sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = 'rain', storm = storm, data_libs = data_libs,
                                    area_mask = None,bbox = bbox_beira, grid_setup = None, writing_mode = 'update',
                                    topo_map = 'beira_dem', res = res, mode = 'rain', tref = tref, tstart = tstart, tstop = tstop)

# now with double precipitation


# 4) RUN ONSHORE SFINCS MODEL
sfincs_pipe.run_sfincs(base_root = r'D:\paper_4\data\sfincs_input\test_base_sfincs_surge', fn_exe = fn_exe) # test_slr100_sfincs_surge




# mod_nr = sfincs_pipe.generate_maps(sfincs_root = r'D:\paper_4\data\sfincs_input\test_base_sfincs_surge', catalog_path = f'D:\paper_4\data\sfincs_input\test_base\hydromt_data.yml', storm = storm, scenario = 'sfincs_surge', mode = 'surge')

mod_nr = SfincsModel(r'D:\paper_4\data\sfincs_input\test_base_sfincs_surge', mode="r")
    # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
mod_nr.read_results()
# mod_nr.write_raster(f"results.hmax", compress="LZW")
# _ = mod_nr.plot_forcing()

da_hmax = mod_nr.results["hmax"].max(['timemax'])
da_hmax = da_hmax.where(da_hmax > 0.1)
fig, ax = mod_nr.plot_basemap(
    fn_out=None,
    figsize=(16, 12),
    variable=da_hmax,
    plot_bounds=False,
    plot_geoms=False,
    bmap="sat",
    zoomlevel=12,
    vmin=0.0,
    vmax=2.0,
    alpha=0.8,
    cmap=plt.cm.Blues,
    cbar_kwargs = {"shrink": 0.6, "anchor": (0, 0)}
)
# ax.set_title(f"SFINCS maximum water depth")
# plt.savefig(join(mod.root, 'figs', 'hmax.png'), dpi=225, bbox_inches="tight")

plt.show()



# open the sfincs_map.nc file and plot the map
ds = xr.open_dataset(f'D:\paper_4\data\quadtree_era5\sfincs_map_spider.nc')
ds_his = xr.open_dataset(f'D:\paper_4\data\quadtree_era5\sfincs_his_era5.nc')
ds_his_tides = xr.open_dataset(f'D:\paper_4\data\quadtree_era5\sfincs_his_tides.nc')
ds_his_spider = xr.open_dataset(f'D:\paper_4\data\quadtree_era5\sfincs_his_spider.nc')
# ds_truth = xr.open_dataset(rf'P:\11209202-tc-risk-analysis-2023\01_Beira_forecasting_TCFF\04_model_runs\20230509_variations_Python_based_n10000\days_before_landfall_1\ensemble09430\sfincs_his.nc')
# ds_truth2 = xr.open_dataset(rf'P:\11209202-tc-risk-analysis-2023\01_Beira_forecasting_TCFF\04_model_runs\20230310_variations_Python_based\tide_only\sfincs_his.nc')
# get maximum value of point_zs for each ds_his and ds_truth
ds_his['point_zs'].max()
# ds_truth['point_zs'].max()

# plot the timeseries for point_zs for station 'station_name' == 10
ds_his['point_zs'].isel(stations = 4).plot(label = 'henrique_era5')
ds_his_tides['point_zs'].isel(stations = 4).plot(label = 'kees_tides_only')
ds_his_spider['point_zs'].isel(stations = 4).plot(label = 'kees_spiderweb')
# ds_truth['point_zs'].isel(stations = 4).plot(label = 'kees2')
# ds_truth2['point_zs'].isel(stations = 4).plot(label = 'kees2_tides_only')
plt.legend()
plt.show()

# display a list of vars of ds_his
list(ds.data_vars)

ds['zsmax'].plot()
plt.show()


# open precip.nc in D:\paper_3\data\sfincs_ini\spectral_nudging\sandy_counter_2_rain_surge
ds2 = xr.open_dataset(r'D:\paper_3\data\sfincs_ini\spectral_nudging\sandy_counter_2_rain_surge\precip.nc')

ds3 = xr.open_dataset(r'D:\paper_4\data\era5\era5_hourly_vars_idai_single_2019_03.nc')

ds4 = xr.open_dataset(r'D:\paper_4\data\sfincs_input\test_forcing\precip_2d.nc')
ds_wind = xr.open_dataset(r'D:\paper_4\data\sfincs_input\test_forcing\wind_2d.nc')
# calculate total wind speed based on  eastward_wind   (time, y, x) float32 northward_wind  (time, y, x) 
ds_wind['wind_speed'] = np.sqrt(ds_wind['eastward_wind']**2 + ds_wind['northward_wind']**2)


# Select a specific time slice
precip = ds_wind['wind_speed'].isel(time=160)
precip = precip.where(precip > 0)
# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
# Plot the data
precip.plot(ax=ax)
# Set a title with the time of the data
ax.set_title(precip.time.values)

# Show the plot
plt.show()

# plot ds4['Precipitation'] timeseries for x and y == 742438.71 7748921.02 
ds4['Precipitation'].sel(x=742438.71, y=7748921.02, method='nearest').plot()
plt.show()

ds_wind['wind_speed'].sel(x=742438.71, y=7748921.02, method='nearest').plot()
plt.show()

import geopandas as gpd
from shapely.geometry import Point

# Provided text data
text_data = """751837.9 7603938.5
696254.9 7719140.0
911627.0 8018433.5
869574.0 7947216.0
692022.6 7807764.5
864218.5 7943624.0
1046236.4 8083667.5
1238897.1 8192773.0
1048013.9 8083620.0
1165472.1 8137639.0
714653.5 7663545.5
698108.6 7807526.0
693118.7 7807692.5
694660.0 7806506.0
697044.1 7804920.5
689725.6 7808857.0
689237.6 7803534.5
694116.6 7801006.0
702011.8 7795151.5
719842.2 7783974.0
696068.2 7808162.0
692322.3 7805136.5
692404.9 7805118.5
692413.1 7805113.0
700518.4 7806124.0
700530.4 7806135.0
691898.9 7806476.5
694115.4 7804354.0
693433.0 7804806.5
701541.6 7806952.0
700697.8 7806128.0
699167.1 7804715.0
697459.9 7803773.0
695752.6 7803851.5
694555.6 7804087.0
692593.2 7804264.0
692024.1 7805107.5
691906.4 7806128.0
692161.5 7808797.0
691769.0 7810151.0
692553.9 7811740.5
694163.1 7812486.0
693535.1 7810563.0
694084.6 7807089.5
692946.4 7805461.0
693888.4 7804833.0
695144.3 7804244.0
695183.5 7803989.0
696341.3 7804558.0
698303.7 7805147.0
699383.0 7806422.5
700089.4 7807678.5
698617.7 7808483.0
696537.5 7807089.5
696478.7 7806089.0
801879.4 7700561.0
800330.0 7701766.0
798780.6 7702971.0
797231.2 7704176.0
795681.8 7705381.5
794132.4 7706586.5
792583.0 7707791.5
791033.6 7708996.5
789484.2 7710201.5
787934.8 7711406.5
786385.4 7712612.0
784836.0 7713817.0
783286.6 7715022.0
781737.2 7716227.0
780187.8 7717432.0
778638.4 7718637.5
777089.0 7719842.5
775539.6 7721047.5
773990.2 7722252.5
772440.8 7723457.5
770891.4 7724662.5
769341.9 7725868.0
767792.6 7727073.0
766243.2 7728278.0
764693.8 7729483.0
763210.1 7730779.0
761726.4 7732075.0
760242.8 7733370.5
758759.1 7734666.5
757275.4 7735962.5
755791.8 7737258.0
754308.1 7738554.0
752824.4 7739850.0
751340.8 7741146.0
749857.1 7742441.5
748373.4 7743737.5
746889.8 7745033.5
745406.1 7746329.5
743922.4 7747625.0
742438.7 7748921.0
740955.1 7750217.0
739471.4 7751513.0
737987.7 7752808.5
736504.0 7754104.5
735020.4 7755400.5
733536.7 7756696.0
732053.0 7757992.0
730749.9 7759390.5
729446.9 7760789.0
728143.8 7762187.5
726840.7 7763586.0
725537.6 7764984.0
724234.5 7766382.5
722931.4 7767781.0
721628.3 7769179.5
720325.2 7770578.0
719022.2 7771976.5
717719.1 7773375.0
716416.0 7774773.5
715112.9 7776171.5
713632.4 7777411.0
712151.8 7778651.0
710671.2 7779890.5
709190.8 7781130.0
707710.2 7782369.5
706229.7 7783609.0
704749.1 7784848.5
703268.6 7786088.0
701788.1 7787327.5
700307.5 7788567.0
698826.9 7789806.5
697346.4 7791046.0
696106.9 7792492.0
694867.4 7793938.0
693627.9 7795384.5
692388.3 7796830.5
691148.8 7798276.5
689909.3 7799722.5
689496.1 7801458.0
689082.9 7803193.5
688669.8 7804928.5
688256.6 7806664.0
687843.4 7808399.5
652803.0 7800536.0
675386.4 7798665.5
661141.4 7843165.5
660502.8 7831180.5
686321.4 7809440.0"""

# Extract coordinates from the text data
coordinates = [tuple(map(float, line.split())) for line in text_data.split('\n')]

# Create Shapely Point geometries
points = [Point(x, y) for x, y in coordinates]

# Create a GeoDataFrame with the correct CRS
gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:32736")  # UTM Zone 36S

# Save to a GeoJSON file
gdf_points.to_file("points.geojson", driver="GeoJSON")

# Display the GeoDataFrame
print(gdf_points)


# Save to a GeoJSON file
gdf_points.to_file("D:\paper_4\data\sfincs_input\polygon.geojson", driver="GeoJSON")

# Display the GeoDataFrame
print(gdf_line)
