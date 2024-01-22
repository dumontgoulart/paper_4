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


mod_nr = SfincsModel(r'D:\paper_4\data\sfincs_input\test_surge_ifs_cf_bc_floodwall', mode="r") #test_rain_gpm
    # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
mod_nr.read_results()
# mod_nr.write_raster(f"results.hmax", compress="LZW")
# _ = mod_nr.plot_forcing()

da_hmax = mod_nr.results["hmax"].max(['timemax'])
da_hmax = da_hmax.where(da_hmax > 0.05)
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

da_hmax.rio.to_raster(rf'D:\paper_4\data\sfincs_output\test\test_surge_ifs_cf_bc.tiff', tiled=True, compress='LZW')

