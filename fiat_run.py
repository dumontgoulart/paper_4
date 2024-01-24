# -*- coding: utf-8 -*-
# First, all required packages must be installed.

from hydromt_fiat.fiat import FiatModel
from hydromt.log import setuplog
from pathlib import Path
import geopandas as gpd
import pandas as pd
import os
import json
import yaml
from hydromt.config import configread
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import rasterio
from rasterio.enums import Resampling
import toml

## this is not really the fiat running part, this is to clean the data that was not working originally.
# Paths
tiff_file_path = r'D:\paper_4\data\vanPanos\FIAT_model_new\Hazard\test_surge_ifs_cf_bc_slr100.tiff'
output_tiff = r'D:\paper_4\data\vanPanos\FIAT_model_new\Hazard\test_surge_ifs_cf_bc_slr100_flip.tiff'

with rasterio.open(tiff_file_path) as src:
    # The transform property returns an affine transformation matrix
    # which maps row/col indexes into coordinates
    transform = src.transform

    # Check if the 6th element of the transform (E) is positive (north up)
    if transform.e > 0:
        print("Latitude is not in the correct order (north up).")
    else:
        print("Latitude order is correct (north up).")


def flip_raster(input_path, output_path):
    with rasterio.open(input_path) as src:
        # Read metadata and data from source
        meta = src.meta.copy()
        data = src.read()

        # Flip the data along the first axis (latitude axis)
        flipped_data = data[:, ::-1, :]

        # Modify the transform
        transform = src.transform
        new_transform = rasterio.Affine(transform.a, transform.b, transform.c,
                                        transform.d, -transform.e, transform.f + transform.e * src.height)

        # Update metadata
        meta.update({"transform": new_transform, "height": flipped_data.shape[1]})

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(flipped_data)

# Flip the raster
flip_raster(tiff_file_path, output_tiff)


##########################################################################################
# 1. Run fiat model TODO: this is not working yet
logger_name = "hydromt_fiat"  # name of the logger
logger = setuplog(logger_name, log_level=10) # setup logg



# Trying to run the model from python
import fiat
from fiat.io import *
from fiat.main import *

FIAT.from_path(r"D:\paper_4\data\vanPanos\FIAT_model_new\settings.toml").run()


config = toml.load(r"D:\paper_4\data\vanPanos\FIAT_model_new\settings.toml")

# Get the 'file' key from the 'hazard' section
scenario = config['hazard']['file'][7:-5]

# 2. Read fiat output and plots

#Load *.csv into dataframe
df_exposure = pd.read_csv(r"D:\paper_4\data\vanPanos\FIAT_model_new\Exposure\exposure_clip3.csv")
# Load exposure geopackage into GeoDataFrame
gdf_exposure =gpd.read_file(r"D:\paper_4\data\vanPanos\FIAT_model_new\Exposure\exposure_clip3.gpkg")
# Merge dataframe with GeoDataFrame
merged_gdf = gdf_exposure.merge(df_exposure, left_on='Object ID', right_on='Object ID', how='inner')
merged_gdf['geometry'] = merged_gdf.geometry.representative_point()



# plot merged_gdf 'Max Potential Damage: Structure'
merged_gdf.plot(column = 'Max Potential Damage: Structure', legend = True)
plt.show()


# Load output geopackage into GeoDataFrame
gdf_output =gpd.read_file(r"D:\paper_4\data\vanPanos\FIAT_model_new\output\spatial.gpkg")
# Merge dataframe with GeoDataFrame
gdf_output['geometry'] = merged_gdf.geometry.representative_point()

# filter for total damage > 0
gdf_output = gdf_output[gdf_output['Total Damage'] > 0]

gdf_output.plot(column = 'Total Damage', legend = True)
# add title
plt.title(f'Total Damage {round(gdf_output["Total Damage"].sum(), 2)}')
plt.show()

# save to geopackage
gdf_output.to_file(rf"D:\paper_4\data\vanPanos\FIAT_model_new\output\fiat_spatial_{scenario}_filter.gpkg", driver="GPKG")




# # File paths
# file_path = rf'D:\paper_4\data\vanPanos\qgis_data\exposure_clip3.csv'
# output_file_path = rf'D:\paper_4\data\vanPanos\qgis_data\exposure_clip3.csv'

# # Read the CSV file
# df = pd.read_csv(file_path, dtype=str)

# # Removing double quotes from "Object name" column
# df['Object Name'] = df['Object Name'].str.replace('"', '')

# # Save the DataFrame to a new CSV file
# df.to_csv(output_file_path, index=False)