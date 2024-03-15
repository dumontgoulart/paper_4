# -*- coding: utf-8 -*-
"""
Prepare files for HydroMT
Created on Tue Oct  9 16:39:47 2022
@author: morenodu
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
import configparser
import yaml
from datetime import datetime
from os.path import join, isfile, isdir, dirname
import subprocess

import hydromt
from hydromt_sfincs import SfincsModel, utils

#general params   
base_folder = 'D:/paper_4/data/sfincs_input/'  #'D:/paper_4/data/version006_quadtree_8km_500m_60s/'
storm = 'idai'
data_libs = ['d:/paper_4/data/data_catalogs/data_catalog_converter.yml', base_folder+rf'/data_deltares_{storm}/data_catalog.yml']

# choose scenario
scenario = 'idai_ifs_rebuild_bc_hist_rain_surge_noadapt' #test_surge_ifs_rebuild_idai_bc test_rain_ifs_cf_bc
tif_file = rf'D:\paper_4\data\sfincs_output\test\{scenario}.tiff'

mod_nr = SfincsModel(base_folder+scenario, data_libs = data_libs, mode="r") #test_rain_gpm
    # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
mod_nr.read_results()
# mod_nr.write_raster(f"results.hmax", compress="LZW")
# _ = mod_nr.plot_forcing()
landmask = mod_nr.data_catalog.get_geodataframe(f"D:\paper_4\data\sfincs_input\data_deltares_idai\osm_landareas.gpkg")

da_hmax = mod_nr.results["hmax"].max(['timemax'])
mask = da_hmax.raster.geometry_mask(landmask)

da_h = mod_nr.results["h"].isel(time=-1)
da_h = da_h.where(da_h > 0.05).where(mask)

da_hmax = da_hmax.where(da_hmax > 0.05).where(mask)

# update attributes for colorbar label later
da_h.attrs.update(long_name="flood depth", unit="m")
# check it's in north-up order
if da_h.y.values[0] < da_h.y.values[-1]:
    # Flip vertically
    da_h = da_h[::-1, :]
    print("Flipped the raster as it was not in north-up order.")
else:
    print("Raster already in north-up order, no flip needed.")

fig, ax = mod_nr.plot_basemap(
    fn_out=None,
    figsize=(16, 12),
    variable=da_h,
    plot_bounds=False,
    plot_geoms=False,
    bmap="sat",
    zoomlevel=14,
    vmin=0.0,
    vmax=5.0,
    alpha=0.8,
    cmap=plt.cm.Blues,
    cbar_kwargs = {"shrink": 0.6, "anchor": (0, 0)}
)
# ax.set_title(f"SFINCS maximum water depth")
# plt.savefig(join(mod.root, 'figs', 'hmax.png'), dpi=225, bbox_inches="tight")
plt.show()

da_hmax.rio.to_raster(tif_file, tiled=True, compress='LZW')


# load this file D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios\idai_ifs_rebuild_bc_hist_rain_surge_noadapt\Flooding\simulations\overland\sfincs_map.nc

sfincs_map = xr.open_dataset(r'D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios\idai_ifs_rebuild_bc_hist_rain_surge_noadapt\Flooding\simulations\overland\sfincs_map.nc')

# now find the "h" for the last time step
sfincs_map_1 = sfincs_map['h'].isel(time=-1)

# plot results
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
sfincs_map_1.plot(ax=ax, cmap='Blues', add_colorbar=True)
ax.set_title('Flood depth [m]')
plt.show()