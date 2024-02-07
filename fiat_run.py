# -*- coding: utf-8 -*-
# First, all required packages must be installed.

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
import shutil

import fiat
from fiat.io import *
from fiat.main import *

##########################################################################################
# 1. Run fiat model 


#### Change toml
def update_hazard_file(toml_file_path, new_hazard_file):
    # Read the existing TOML file
    with open(toml_file_path, 'r') as file:
        data = toml.load(file)
    
    # Update the "file" field under the "hazard" section
    if 'hazard' in data:
        data['hazard']['file'] = new_hazard_file
    else:
        print("The 'hazard' section is missing in the TOML file.")
    
    # Write the updated data back to the TOML file
    with open(toml_file_path, 'w') as file:
        toml.dump(data, file)
    print(f"Updated 'hazard' file to: {new_hazard_file}")


def run_fiat_scenario(fiat_root_folder, new_hazard_file, obtain_scenarios=None):
    """
    Updates the hazard file in the FIAT model configuration, runs the FIAT model,
    and post-processes the output to filter and save geospatial data based on total damage.

    :param fiat_root_folder: Root folder for the FIAT model and related files.
    :param new_hazard_file: New hazard file to update in the model configuration.
    """
    # Define file paths
    toml_file_path = rf'{fiat_root_folder}\settings.toml'
    output_file_path = rf'{fiat_root_folder}\output\spatial.gpkg'

    #0 Obtain scenarios
    if obtain_scenarios:
        shutil.copy(obtain_scenarios, rf'{fiat_root_folder}\Hazard')

    # 1 Prepare the toml file - update hazard for scenario
    update_hazard_file(toml_file_path, f'hazard/{new_hazard_file}')

    # 2 RUN FIAT (Assuming FIAT.from_path is a valid command, replace with actual implementation if necessary)
    FIAT.from_path(toml_file_path).run()

    # 3 Post process the output
    config = toml.load(toml_file_path)
    scenario = config['hazard']['file'][7:-5]  # Assumes the file path format; adjust slicing as needed
    gdf_output = gpd.read_file(output_file_path)
    gdf_output = gdf_output[gdf_output['Total Damage'] > 0]
    gdf_output.to_file(rf"{fiat_root_folder}\output\fiat_spatial_{scenario}.gpkg", driver="GPKG")


fiat_root_folder = r'D:\paper_4\data\vanPanos\FIAT_model_new'

adapt_scenarios = ['idai_ifs_rebuild_bc_3c-hightide_rain_surge_hold',
             'idai_ifs_rebuild_bc_3c-hightide_rain_surge_retreat', 
             'idai_ifs_rebuild_bc_3c-hightide_rain_surge_noadapt']

for adapt_scenario in adapt_scenarios:
    run_fiat_scenario(fiat_root_folder, f'{adapt_scenario}.tiff', obtain_scenarios=rf'D:\paper_4\data\sfincs_output\test\{adapt_scenario}.tiff')


# # Paths
# fiat_root_folder = r'D:\paper_4\data\vanPanos\FIAT_model_new'
# toml_file_path = rf'{fiat_root_folder}\settings.toml'
# output_file_path = rf'{fiat_root_folder}\output\spatial.gpkg'
# new_hazard_file = 'idai_ifs_rebuild_bc_hist_rain_surge_retreat.tiff'

# # 1 Prepare the toml file - update hazard for scenario
# update_hazard_file(toml_file_path, 'hazard/'+new_hazard_file)

# # 2 RUN FIAT
# FIAT.from_path(toml_file_path).run()
# # you can also run normally by using on cmd: fiat run FIAT_model_new/settings.toml

# # 3 Post process the output
# config = toml.load(toml_file_path)
# # Get the 'file' key from the 'hazard' section
# scenario = config['hazard']['file'][7:-5]
# # Load output geopackage into GeoDataFrame
# gdf_output = gpd.read_file(output_file_path)
# # filter for total damage > 0
# gdf_output = gdf_output[gdf_output['Total Damage'] > 0]

# # save to geopackage
# gdf_output.to_file(rf"{fiat_root_folder}\output\fiat_spatial_{scenario}.gpkg", driver="GPKG")



# gdf_output.plot(column = 'Total Damage', legend = True)
# # add title
# plt.title(f'Total Damage {round(gdf_output["Total Damage"].sum(), 2)}')
# plt.show()


# #Load *.csv into dataframe for base geometry
# df_exposure = pd.read_csv(r"D:\paper_4\data\vanPanos\FIAT_model_new\Exposure\exposure_clip3.csv")
# # Load exposure geopackage into GeoDataFrame
# gdf_exposure =gpd.read_file(r"D:\paper_4\data\vanPanos\FIAT_model_new\Exposure\exposure_clip3.gpkg")
# # Merge dataframe with GeoDataFrame
# merged_gdf = gdf_exposure.merge(df_exposure, left_on='Object ID', right_on='Object ID', how='inner')
# merged_gdf['geometry'] = merged_gdf.geometry.representative_point()

# UNCOMMENT THIS IF YOU NEED TO CORRECT THE CSV FILE
# # File paths - change to your file paths
# file_path = rf'D:\paper_4\data\vanPanos\qgis_data\exposure_clip4.csv'
# output_file_path = rf'D:\paper_4\data\vanPanos\qgis_data\exposure_clip4.csv'

# # Read the CSV file
# df = pd.read_csv(file_path, dtype=str)

# # Removing double quotes from "Object name" column
# df['Object Name'] = df['Object Name'].str.replace('"', '')

# # Save the DataFrame to a new CSV file
# df.to_csv(output_file_path, index=False)

# uncomment this to correct the tiff file
# ## this is not really the fiat running part, this is to clean the data that was not working originally.
# # Paths
# tiff_file_path = r'D:\paper_4\data\vanPanos\FIAT_model_new\Hazard\test_surge_ifs_cf_bc_slr100.tiff'
# output_tiff = r'D:\paper_4\data\vanPanos\FIAT_model_new\Hazard\test_surge_ifs_cf_bc_slr100_flip.tiff'

# with rasterio.open(tiff_file_path) as src:
#     # The transform property returns an affine transformation matrix
#     # which maps row/col indexes into coordinates
#     transform = src.transform

#     # Check if the 6th element of the transform (E) is positive (north up)
#     if transform.e > 0:
#         print("Latitude is not in the correct order (north up).")
#     else:
#         print("Latitude order is correct (north up).")


# def flip_raster(input_path, output_path):
#     with rasterio.open(input_path) as src:
#         # Read metadata and data from source
#         meta = src.meta.copy()
#         data = src.read()

#         # Flip the data along the first axis (latitude axis)
#         flipped_data = data[:, ::-1, :]

#         # Modify the transform
#         transform = src.transform
#         new_transform = rasterio.Affine(transform.a, transform.b, transform.c,
#                                         transform.d, -transform.e, transform.f + transform.e * src.height)

#         # Update metadata
#         meta.update({"transform": new_transform, "height": flipped_data.shape[1]})

#         with rasterio.open(output_path, 'w', **meta) as dst:
#             dst.write(flipped_data)

# # Flip the raster
# flip_raster(tiff_file_path, output_tiff)
