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
scenario = 'snellius_idai_ifs_rebuild_bc_3c_rain_surge' #test_surge_ifs_rebuild_idai_bc test_rain_ifs_cf_bc
tif_file = rf'D:\paper_4\data\sfincs_output\test\{scenario}.tiff'

mod_nr = SfincsModel(base_folder+scenario, data_libs = data_libs, mode="r") #test_rain_gpm
    # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
mod_nr.read_results()
# mod_nr.write_raster(f"results.hmax", compress="LZW")
# _ = mod_nr.plot_forcing()

gswo = mod_nr.data_catalog.get_rasterdataset("gswo", geom=mod_nr.region, buffer=10)
gswo_mask = gswo.raster.reproject_like(mod_nr.grid, method="max") <= 5

da_hmax = mod_nr.results["hmax"].max(['timemax'])
da_hmax = da_hmax.where(gswo_mask).where(da_hmax > 0.05)
# update attributes for colorbar label later
da_hmax.attrs.update(long_name="flood depth", unit="m")
# check it's in north-up order
if da_hmax.y.values[0] < da_hmax.y.values[-1]:
    # Flip vertically
    da_hmax = da_hmax[::-1, :]
    print("Flipped the raster as it was not in north-up order.")
else:
    print("Raster already in north-up order, no flip needed.")

fig, ax = mod_nr.plot_basemap(
    fn_out=None,
    figsize=(16, 12),
    variable=da_hmax,
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