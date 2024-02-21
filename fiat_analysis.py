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
import numpy as np
import fiat
from fiat.io import *
from fiat.main import *


# general setup for plots
plt.rcParams['font.family'] = 'Roboto'
singlecol = 8.3 * 0.393701
doublecol = 14 * 0.393701
##########################################################################################

# Assuming the alignment function needs to handle missing geometries appropriately
def align_gdf(gdf, all_ids, master_geometry_dict):
    # Ensure the GDF includes all 'Object IDs', filling missing ones with default values
    aligned_df = pd.DataFrame(all_ids, columns=['Object ID']).merge(
        gdf.drop(columns='geometry'),
        on='Object ID',
        how='left'
    ).fillna({'Total Damage': 0})

    # Assign geometries from the master_geometry_dict based on 'Object ID'
    aligned_df['geometry'] = aligned_df['Object ID'].map(master_geometry_dict)

    # Convert back to GeoDataFrame
    aligned_gdf = gpd.GeoDataFrame(aligned_df, geometry='geometry', crs=gdf.crs)
    
    return aligned_gdf

##########################################################################################

fiat_output_path = r'D:\paper_4\data\vanPanos\FIAT_model_new\output'
sim = 'idai_ifs_rebuild_bc_'

# scenarios
climate_scenarios = ['hist', '3c', 'hightide', '3c-hightide']
adapt_scenarios = ['noadapt', 'hold','retreat']

# Generate all combinations of climate and adaptation scenarios
all_adapt_scenarios = [f"{sim}{climate}_rain_surge_{adapt}" for climate in climate_scenarios for adapt in adapt_scenarios]


# Initialize an empty list to store each GeoDataFrame
gdf_list = []

# Loop through each combination of scenarios
for climate in climate_scenarios:
    for adapt in adapt_scenarios:
        # Construct the file name based on the scenario
        file_name = rf'{fiat_output_path}\fiat_spatial_hmax_{sim}{climate}_rain_surge_{adapt}.gpkg'
        
        # Load the .gpkg file
        gdf = gpd.read_file(file_name)
        
        # Add climate and adaptation scenario as columns
        gdf['climate_scenario'] = climate
        gdf['adapt_scenario'] = adapt
        
        # Append to the list
        gdf_list.append(gdf)

# Concatenate all GeoDataFrames into one
all_gdf = pd.concat(gdf_list, ignore_index=True)


# Step 1: Aggregate the data
agg_data = all_gdf.groupby(['climate_scenario', 'adapt_scenario'])['Total Damage'].sum().reset_index()

# Step 2: Create the bar charts
# Define the order of adaptation scenarios and climate scenarios for consistency in plotting
adapt_order = ['noadapt', 'hold', 'retreat']
climate_order = ['hist', '3c', 'hightide', '3c-hightide']

# plot 2 - all on the same plot
colors = ['salmon', '#9A607F', '#B4BA39']  # Colors for each adaptation scenario

# Number of climate scenarios
n_climate = len(climate_scenarios)
# Width of the bars
bar_width = 0.15

# Set positions of the bars for each adaptation scenario
positions = np.arange(len(climate_scenarios))  # Base positions

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, adapt in enumerate(adapt_order):
    # Calculate offset for each group based on adaptation scenario index
    offset = (i - np.floor(len(adapt_scenarios) / 2)) * bar_width
    # Filter data for the current adaptation scenario
    adapt_data = agg_data[agg_data['adapt_scenario'] == adapt]
    # Ensure the data is in the order of climate scenarios
    adapt_data = adapt_data.set_index('climate_scenario').reindex(climate_scenarios).reset_index()
    # Plot
    ax.bar(positions + offset, adapt_data['Total Damage'], width=bar_width, label=adapt, color=colors[i], zorder=2)

# Formatting the plot
ax.set_xlabel('Climate Scenario')
ax.set_ylabel('Total Damage [$]')
ax.set_title('Total Damage of Idai storylines in Beira', fontsize=14, fontweight='bold', loc='left')
ax.set_xticks(positions)
ax.set_xticklabels(climate_scenarios)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.8, 1.1), ncol=3, fontsize=12)

plt.show()

# figure 3 - percentage changes
pivot_data = agg_data.pivot(index='climate_scenario', columns='adapt_scenario', values='Total Damage')

# Calculate percentages relative to 'noadapt'
pivot_data['hold_pct'] = (pivot_data['hold'] / pivot_data['noadapt']) * 100 - 100
pivot_data['retreat_pct'] = (pivot_data['retreat'] / pivot_data['noadapt']) * 100 - 100
pivot_data['noadapt_pct'] = 100 - 100 # Baseline

# Prepare data for plotting
percent_data = pd.DataFrame({
    'climate_scenario': pivot_data.index,
    'noadapt': pivot_data['noadapt_pct'],
    'hold': pivot_data['hold_pct'],
    'retreat': pivot_data['retreat_pct']
}).melt(id_vars='climate_scenario', var_name='adapt_scenario', value_name='Damage Percent')
# drop rows where adapt_scenario is noadapt
percent_data = percent_data[percent_data['adapt_scenario'] != 'noadapt']


# Plotting
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
colors_diff = ['#9A607F', '#B4BA39']  # Colors for each adaptation scenario
# Add thin gray horizontal lines at intervals of 20
ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set positions of the bars for each adaptation scenario
positions = np.arange(len(climate_scenarios))  # Base positions

for i, adapt in enumerate(adapt_order[1:3]):
    # Calculate offset for each group based on adaptation scenario index
    offset = (i - np.floor(len(adapt_order[1:3]) / 2)) * bar_width
    # Filter data for the current adaptation scenario
    adapt_data = percent_data[percent_data['adapt_scenario'] == adapt]
    # Ensure the data is in the order of climate scenarios
    adapt_data = adapt_data.set_index('climate_scenario').reindex(climate_scenarios).reset_index()
    # Plot
    ax.bar(positions + offset, adapt_data['Damage Percent'], width=bar_width, label=adapt, color=colors_diff[i], zorder=2)
# add a dahsed horizontal line at 0
# Formatting the plot
ax.set_xlabel('Climate Scenario')
ax.set_ylabel('Damage change (%)')
ax.set_title('Damage wrt No Adaptation', fontsize=14, fontweight='bold', loc='left')
ax.set_xticks(positions)
ax.set_xticklabels(climate_scenarios)
ax.set_ylim(-100, 10)  # Adjust as needed based on your data
ax.legend(frameon=False,  bbox_to_anchor=(0.85, 1.0), ncol=2, loc='upper center', fontsize=12)

plt.show()


# now load hazards to analyse
# Load the hazard data
hazard_folder_path = r'D:\paper_4\data\vanPanos\FIAT_model_new\Hazard'


# Initialize a dictionary to store the loaded data
raster_data = {}

# Loop through each combination of scenarios
for climate in climate_scenarios:
    for adapt in adapt_scenarios:
        # Construct the file name based on the scenario
        file_name = f'{hazard_folder_path}/hmax_{sim}{climate}_rain_surge_{adapt}.tiff'
        
        # Load the .tiff file using rasterio
        with rasterio.open(file_name) as src:
            # Read the raster data
            data = src.read(1)  # Reads the first band; adjust as needed
            
            # Optional: mask out no data values if necessary
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            
            # Store the data in the dictionary
            key = f'{climate}_{adapt}'
            raster_data[key] = data


# plot a pdf of raster_data['hist_noadapt'] 
            

# Step 1: Calculate the average value for each raster dataset
avg_values = {}
for key, data in raster_data.items():
    # Calculate the average, excluding NaN values
    avg_value = np.nansum(data)
    avg_values[key] = avg_value

# Step 2: Prepare the data for plotting
# Convert the avg_values dictionary into a DataFrame for easier plotting
df = pd.DataFrame(list(avg_values.items()), columns=['Scenario', 'SumValue'])
df['ClimateScenario'], df['AdaptScenario'] = zip(*df['Scenario'].str.split('_'))

# Now, sort the DataFrame by these columns to ensure the order is respected in the pivot
df_sorted = df.sort_values(['ClimateScenario', 'AdaptScenario'])

# Step 2: Pivot the sorted DataFrame
pivot_df = df_sorted.pivot(index="ClimateScenario", columns="AdaptScenario", values="SumValue")

# Optional: Reindex the pivot table to ensure the order (this step may be redundant if sorting was effective)
pivot_df = pivot_df.reindex(index=climate_order, columns=adapt_order)

# Step 3: Plot the data
pivot_df.plot(kind='bar', figsize=(10, 6), width=0.4, color = colors, zorder=2)
ax = plt.gca()
ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title('Total inundation volume per scenario', fontsize=14, fontweight='bold', loc='left')
plt.ylabel('Total volume (m3)')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(0.75, 1.1), loc='upper center', frameon=False, ncol=3)
plt.tight_layout()
plt.show()


##########################################################################################
# plot maps

from rasterio.plot import show
from matplotlib.colors import Normalize, Colormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
import contextily as ctx

tiff_paths = [
    rf'D:\paper_4\data\sfincs_output\snellius_idai\hmax_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.tiff',
    rf'D:\paper_4\data\sfincs_output\snellius_idai\hmax_idai_ifs_rebuild_bc_hightide_rain_surge_noadapt.tiff',
    rf'D:\paper_4\data\sfincs_output\snellius_idai\hmax_idai_ifs_rebuild_bc_3c_rain_surge_noadapt.tiff',
    rf'D:\paper_4\data\sfincs_output\snellius_idai\hmax_idai_ifs_rebuild_bc_3c-hightide_rain_surge_noadapt.tiff'
]

landareas_path = rf'D:\paper_4\data\sfincs_input\data_deltares_idai\osm_landareas.gpkg'
gdf_landareas = gpd.read_file(landareas_path)
# change crs to match the tiff
gdf_landareas = gdf_landareas.to_crs('EPSG:32736')

# Define colorbar boundaries
boundaries = np.linspace(0, 3, 11)  # 10 intervals from 0 to 3
norm = BoundaryNorm(boundaries, ncolors=256)

# Create a figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(21, 21), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it
# Loop through each TIFF path and plot it
for ax, tiff_path in zip(axes, tiff_paths):
    with rasterio.open(tiff_path) as src:
        # Ensure GeoDataFrame is in the same CRS as the raster
        if gdf_landareas.crs != src.crs:
            gdf_to_plot = gdf_landareas.to_crs(src.crs)
        else:
            gdf_to_plot = gdf_landareas
        
        # Plot the GeoPackage data as basemap
        gdf_to_plot.plot(ax=ax, color='darkgray')  # Adjust color as needed

        # Plot the raster data
        show(src, ax=ax, cmap='Blues', norm=norm, zorder=2)

        # Set the axis limits to the raster bounds
        ax.set_xlim([src.bounds.left, src.bounds.right-2000])
        ax.set_ylim([src.bounds.bottom+1200, src.bounds.top-2000])

        # Optional: Set a title for each subplot based on the TIFF file name or another identifier
        ax.set_title(tiff_path.split('_')[-4])

# remove space between the subplots
plt.subplots_adjust(wspace=-0.40, hspace=0.01)
# Adjust the position of the color bar to not overlap with the subplots
fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.88, 0.15, 0.015, 0.7])
# Create a common color bar for all subplots
cb = ColorbarBase(cbar_ax, cmap='Blues', norm=norm, boundaries=boundaries, ticks=boundaries, spacing='proportional', orientation='vertical')
cb.set_label('Inundation Depth (m)')
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

# plt.tight_layout()
plt.show()

# Function to calculate raster difference
def calculate_raster_difference(path_a, path_b):
    with rasterio.open(path_a) as src_a:
        data_a = src_a.read(1).astype(float)  # Ensure the data is in float for NaN handling
        profile = src_a.profile
        # Replace NA values with 0 only where data_b is not NA
        data_a_masked = np.where(~np.isnan(data_a), data_a, 0)
        
    with rasterio.open(path_b) as src_b:
        data_b = src_b.read(1).astype(float)  # Ensure the data is in float for NaN handling
        # Replace NA values with 0 only where data_a is not NA
        data_b_masked = np.where(~np.isnan(data_b), data_b, 0)
        
    # Calculate the difference where both have data
    difference = np.where(~np.isnan(data_a) & ~np.isnan(data_b), data_a - data_b, 
                          # Where data_a is NA and data_b is not
                          np.where(np.isnan(data_a) & ~np.isnan(data_b), -data_b, 
                                   # Where data_b is NA and data_a is not
                                   np.where(~np.isnan(data_a) & np.isnan(data_b), data_a, 
                                            np.nan)))  # Keep as NA where both are NA
    
    return difference, profile



# Calculate the differences
diff_2_minus_0, profile = calculate_raster_difference(tiff_paths[2], tiff_paths[0])
diff_3_minus_1, _ = calculate_raster_difference(tiff_paths[3], tiff_paths[1])
# Original Rasters Color Range
vmin_orig, vmax_orig = 0, 3

# Difference Rasters Color Range
vmin_diff, vmax_diff = -2, 2

fig, axes = plt.subplots(2, 2, figsize=(21, 21), sharex='col', sharey='row')
axes = axes.flatten()
# Plot the original raster data for the top two subplots
for i, tiff_path in enumerate(tiff_paths[:2]):
    with rasterio.open(tiff_path) as src:
        gdf_landareas.plot(ax=axes[i], color='darkgray')
        show(src, ax=axes[i], cmap='Blues', norm=Normalize(vmin=vmin_orig, vmax=vmax_orig), zorder=2)
        axes[i].set_title(tiff_path.split('_')[-4])

                # Set the axis limits to the raster bounds
        axes[i].set_xlim([src.bounds.left, src.bounds.right-2000])
        axes[i].set_ylim([src.bounds.bottom+1200, src.bounds.top-2000])

# Plot the differences for the bottom two subplots
diff_arrays = [diff_2_minus_0, diff_3_minus_1]
for i, diff_array in enumerate(diff_arrays, start=2):
    title_appendix = '3C - Hist' if i == 2 else '3C-Hightide - Hightide'
    gdf_landareas.plot(ax=axes[i], color='darkgray')
    im = axes[i].imshow(diff_array, cmap='BrBG_r', norm=Normalize(vmin=vmin_diff, vmax=vmax_diff), zorder=2,
                        extent=[profile['transform'][2], profile['transform'][2] + profile['transform'][0] * profile['width'],
                                profile['transform'][5] + profile['transform'][4] * profile['height'], profile['transform'][5]])
    axes[i].set_title(f'{title_appendix}')

    # Set the axis limits to the raster bounds
    axes[i].set_xlim([src.bounds.left, src.bounds.right-2000])
    axes[i].set_ylim([src.bounds.bottom+1200, src.bounds.top-2000])

# Adjust layout for colorbars
plt.subplots_adjust(wspace=-0.5, hspace=0.1, right=0.95)

# Add colorbars
# Original rasters colorbar
cbar_ax_orig = fig.add_axes([0.88, 0.53, 0.015, 0.35])  # Adjust these values as needed
cb_orig = ColorbarBase(cbar_ax_orig, cmap='Blues', norm=Normalize(vmin=vmin_orig, vmax=vmax_orig), orientation='vertical')
cb_orig.set_label('Inundation Depth (m)')

# Difference rasters colorbar
cbar_ax_diff = fig.add_axes([0.88, 0.1, 0.015, 0.35])  # Adjust these values as needed
cb_diff = ColorbarBase(cbar_ax_diff, cmap='BrBG_r', norm=Normalize(vmin=vmin_diff, vmax=vmax_diff), orientation='vertical')
cb_diff.set_label('Inundation Depth Difference (m)')

plt.show()




##########################################################################################
# NOW FOR THE IMPACT VECTORS

# Define your GPKG paths
gpkg_paths = [
    rf'D:\paper_4\data\vanPanos\FIAT_model_new\output\fiat_spatial_hmax_idai_ifs_rebuild_bc_hist_rain_surge_noadapt.gpkg',
    rf'D:\paper_4\data\vanPanos\FIAT_model_new\output\fiat_spatial_hmax_idai_ifs_rebuild_bc_hightide_rain_surge_noadapt.gpkg',
    rf'D:\paper_4\data\vanPanos\FIAT_model_new\output\fiat_spatial_hmax_idai_ifs_rebuild_bc_3c_rain_surge_noadapt.gpkg',
    rf'D:\paper_4\data\vanPanos\FIAT_model_new\output\fiat_spatial_hmax_idai_ifs_rebuild_bc_3c-hightide_rain_surge_noadapt.gpkg'
]

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(22, 22), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it
# Loop through each GPKG path and plot it
for ax, gpkg_path in zip(axes, gpkg_paths):
    # Load the current GeoPackage
    gdf = gpd.read_file(gpkg_path)

    # Set the axis limits to the raster bounds
    ax.set_xlim([src.bounds.left, src.bounds.right-2000])
    ax.set_ylim([src.bounds.bottom+1200, src.bounds.top-2000])

    # gdf_to_plot.plot(ax=ax, color='lightgray')  # Adjust color as needed
    ctx.add_basemap(ax, crs=gdf.crs.to_epsg(), alpha=1, source=ctx.providers.CartoDB.PositronNoLabels, zorder=1)
    gdf.plot(ax=ax, cmap='Reds', markersize = 1, vmin = 0, vmax = 1386549.33, zorder =2 )  # Adjust color and edgecolor as needed #edgecolor='black'
    
    ax.set_title(gpkg_path.split('_')[-4])  # Adjust as needed for a meaningful title

# Adjust layout
plt.subplots_adjust(wspace=-0.4, hspace=0.1)

# Create a colormap scalar mappable object for the colorbar
norm = Normalize(vmin=0, vmax=1386549.33)
sm = ScalarMappable(norm=norm, cmap='Reds')

# Add a colorbar to the right of the subplots
cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])  # Adjust these values as needed
fig.colorbar(sm, cax=cbar_ax).set_label('Total Damage')

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


# differences
run_name = '3c-hightide'
gpkg_path_no_adapt = rf'D:\paper_4\data\vanPanos\FIAT_model_new\output\fiat_spatial_hmax_idai_ifs_rebuild_bc_{run_name}_rain_surge_noadapt.gpkg'
gpkg_path_hold = rf'D:\paper_4\data\vanPanos\FIAT_model_new\output\fiat_spatial_hmax_idai_ifs_rebuild_bc_{run_name}_rain_surge_hold.gpkg'
gpkg_path_retreat = rf'D:\paper_4\data\vanPanos\FIAT_model_new\output\fiat_spatial_hmax_idai_ifs_rebuild_bc_{run_name}_rain_surge_retreat.gpkg'

gdf_no_adapt = gpd.read_file(gpkg_path_no_adapt)
gdf_hold = gpd.read_file(gpkg_path_hold)
gdf_retreat = gpd.read_file(gpkg_path_retreat)

unique_ids = pd.Series(pd.concat([gdf_no_adapt['Object ID'], gdf_hold['Object ID'], gdf_retreat['Object ID']]).unique(), name='Object ID')

# Combine 'Object ID' from all GeoDataFrames to find the unique set
all_ids = pd.Series(pd.concat([gdf_no_adapt['Object ID'], gdf_hold['Object ID'], gdf_retreat['Object ID']]).unique(), name='Object ID')

# First, create a master dictionary of 'Object IDs' to geometries
master_geometry_dict = {}
for gdf in [gdf_no_adapt, gdf_hold, gdf_retreat]:
    for object_id, geometry in zip(gdf['Object ID'], gdf.geometry):
        master_geometry_dict[object_id] = geometry

# Align GDFs
gdf_no_adapt_aligned = align_gdf(gdf_no_adapt, all_ids, master_geometry_dict)
gdf_hold_aligned = align_gdf(gdf_hold, all_ids, master_geometry_dict)
gdf_retreat_aligned = align_gdf(gdf_retreat, all_ids, master_geometry_dict)

diff_gdf_1 = gdf_no_adapt_aligned.copy()
diff_gdf_1['Total Damage'] = (gdf_hold_aligned['Total Damage'] / gdf_no_adapt_aligned['Total Damage'])*100
# drop the 0 values in total damage

diff_gdf_2 = gdf_no_adapt_aligned.copy()
diff_gdf_2['Total Damage'] = (gdf_retreat_aligned['Total Damage'] / gdf_no_adapt_aligned['Total Damage'])*100

diff_gdf_3 = gdf_hold_aligned.copy()
diff_gdf_3['Total Damage'] = (gdf_retreat_aligned['Total Damage'] / gdf_hold_aligned['Total Damage'])*100


# Determine global min and max for consistent color scaling across subplots
vmin = 0
vmax = 200
# Define the original colormap
original_cmap = plt.cm.RdBu_r
# Create a custom colormap with white at the center (value 100)
colors = original_cmap(np.linspace(0, 1, 256))
min_white = np.abs(np.linspace(vmin, vmax, 256) - 95).argmin()  # Find index for 99
max_white = np.abs(np.linspace(vmin, vmax, 256) - 105).argmin()  # Find index for 101
colors[min_white:max_white + 1] = [1, 1, 1, 1]  # Set colors in this range to white
custom_cmap = LinearSegmentedColormap.from_list('CustomRdBu_r', colors)

boundaries = np.linspace(vmin, vmax, 21)  # Creates 10 intervals between 0 and 200
# Create a normalization based on these boundaries
norm = BoundaryNorm(boundaries, ncolors=256, clip=True)

# Figure comparing the adaptations to the no adapatation scenario
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
# Base plotting for all points
gdf_to_plot.plot(ax=axes[0], color='darkgray', zorder=1)  # Adjust color as needed
gdf_to_plot.plot(ax=axes[1], color='darkgray', zorder=1)  # Adjust color as needed

diff_gdf_1.plot(ax=axes[0], column='Total Damage', cmap=custom_cmap, norm=norm, 
                markersize=10, linewidth=0.0, zorder=2)
axes[0].set_title('Hold - No Adapt')
axes[0].axis('off')

diff_gdf_2.plot(ax=axes[1], column='Total Damage', cmap=custom_cmap, norm=norm,
                markersize=10, linewidth=0.0, zorder=2)
axes[1].set_title('Retreat - No Adapt')
axes[1].axis('off')
#remove horizontal space between subplots
plt.subplots_adjust(wspace=0.0)

# ctx.add_basemap(ax=axes[0], crs=gdf.crs.to_epsg(), alpha=0.5, source=ctx.providers.CartoDB.PositronNoLabels, zorder=1)
# ctx.add_basemap(ax=axes[1], crs=gdf.crs.to_epsg(), alpha=0.5, source=ctx.providers.CartoDB.PositronNoLabels, zorder=1)
# Set the axis limits to the raster bounds
for ax in axes:
    ax.set_xlim([src.bounds.left, src.bounds.right-2000])
    ax.set_ylim([src.bounds.bottom+1200, src.bounds.top-2000])

# Create and position the color bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.86, 0.15, 0.015, 0.7])
sm = ScalarMappable(norm=norm, cmap=custom_cmap)
fig.colorbar(sm, cax=cbar_ax, boundaries=boundaries[::2], ticks=boundaries[::2], orientation='vertical').set_label('Change in damage (%)')
plt.show()

# Comparison the two adaptation scenarios
fig, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
# Base plotting for all points
gdf_to_plot.plot(ax=axes, color='darkgray', zorder=1)  # Adjust color as needed

diff_gdf_3.plot(ax=axes, column='Total Damage', cmap=custom_cmap, norm=norm, 
                markersize=10, linewidth=0.0, zorder=2)
axes.set_title('Retreat - Hold')
axes.axis('off')

axes.set_xlim([src.bounds.left, src.bounds.right-2000])
axes.set_ylim([src.bounds.bottom+1200, src.bounds.top-2000])

# Create and position the color bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
sm = ScalarMappable(norm=norm, cmap=custom_cmap)
fig.colorbar(sm, cax=cbar_ax, boundaries=boundaries[::2], ticks=boundaries[::2], orientation='vertical').set_label('Change in damage (%)')

plt.show()


##########################################################################################
# load populations for each scenario
sim = 'idai_ifs_rebuild_bc_'
climate_scenarios = ['hist', '3c', 'hightide', '3c-hightide'] #
adapt_scenarios = ['noadapt', 'retreat', 'hold'] #

# Generate all combinations of climate and adaptation scenarios
all_adapt_scenarios = [f"{sim}{climate}_rain_surge_{adapt}" for climate in climate_scenarios for adapt in adapt_scenarios]

root_fa_output = rf'D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios'

selected_metrics = ['TotalDamageEvent', 'PeopleAffected10cm', 'PeopleAffected30cm',
                    'PeopleAffected50cm', 'PeopleAffected1m']

# Initialize an empty DataFrame to concatenate all CSV files
all_infometrics_df = pd.DataFrame()

for scenario in all_adapt_scenarios:
    # Define the path pattern for the CSV files
    infometrics_csv_path = rf'{root_fa_output}\{scenario}\infometrics_{scenario}.csv'    
    # Use glob to find all files matching the pattern
    infometrics = pd.read_csv(infometrics_csv_path)
    # pivot the table

    filtered_df = infometrics[infometrics['Unnamed: 0'].isin(selected_metrics)].drop(columns=['Unnamed: 0','Description','Show In Metrics Table'])

    filtered_df = filtered_df.pivot_table(columns='Long Name', values='Value').reset_index().drop(columns=['index'])
            
    filtered_df['clim_scen'] = scenario.split('_')[-4]
    filtered_df['adapt_scen'] = scenario.split('_')[-1] 
        
        # Concatenate the current dataframe to the main dataframe
    all_infometrics_df = pd.concat([all_infometrics_df, filtered_df], ignore_index=True)


# Step 1: Aggregate the data for 'People Affected above 150 cm'
agg_info = all_infometrics_df.groupby(['clim_scen', 'adapt_scen'])['People Affected above 150 cm'].sum().reset_index()

# Step 2: Create the bar charts
adapt_order = ['noadapt', 'hold', 'retreat']
climate_order = ['hist', '3c', 'hightide', '3c-hightide']
colors = ['salmon', 'lightblue', 'lightgreen']  # Colors for each adaptation scenario

# Number of climate scenarios
n_climate = len(climate_order)
# Width of the bars
bar_width = 0.15

# Set positions of the bars for each adaptation scenario
positions = np.arange(n_climate)  # Base positions

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, adapt in enumerate(adapt_order):
    # Calculate offset for each group based on adaptation scenario index
    offset = (i - np.floor(len(adapt_order) / 2)) * bar_width
    # Filter data for the current adaptation scenario
    adapt_data = agg_info[agg_info['adapt_scen'] == adapt]
    # Ensure the data is in the order of climate scenarios
    adapt_data = adapt_data.set_index('clim_scen').reindex(climate_order).reset_index()
    # Plot
    ax.bar(positions + offset, adapt_data['People Affected above 150 cm'], width=bar_width, label=adapt, color=colors[i])

# Formatting the plot
ax.set_xlabel('Climate Scenario')
ax.set_ylabel('People Affected above 150 cm')
ax.set_title('People Affected Above 150 cm by Climate and Adaptation Scenarios')
ax.set_xticks(positions)
ax.set_xticklabels(climate_order)
ax.legend(title='Adaptation Scenario', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


# bar plot only for noadapt
noadapt_data = all_infometrics_df[all_infometrics_df['adapt_scen'] == 'noadapt']
# drop column 'People Affected up 15 cm'
noadapt_data = noadapt_data.drop(columns=['People Affected up 15 cm'])

# Step 2: Create the stacked bar chart
climate_order = ['hist', '3c', 'hightide', '3c-hightide']
population_metrics = [
    'People Affected between 15 to 50 cm',
    'People Affected between 50 and 150 cm',
    'People Affected above 150 cm'
]

# Sort noadapt_data based on the order of climate scenarios
noadapt_data = noadapt_data.set_index('clim_scen').reindex(climate_order).reset_index()
blue_colors = ['#a9cce3', '#5dade2', '#2e86c1']

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
tick_increment = 50000
ax.set_yticks(np.arange(0, noadapt_data['People Affected between 50 and 150 cm'].max()*1.1 + tick_increment, tick_increment))

ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
# Define the bottom of the stack
bottom = np.zeros(len(climate_order))

# Plot the bars with the blue color gradient
for i, metric in enumerate(population_metrics):
    ax.bar(
        noadapt_data['clim_scen'],
        noadapt_data[metric],
        bottom=bottom,
        color=blue_colors[i],
        width=0.2, 
        label=metric,
        edgecolor='white',
        zorder=2,
    )
    # Update the bottom position for the next segment of the stack
    bottom += noadapt_data[metric].values

# Formatting the plot
# ax.set_xlabel('Climate Scenario')
ax.set_ylabel('Population exposure')
ax.set_title('Exposed people for No-Adaptation Scenario', fontsize=14, fontweight='bold', loc='left')
ax.legend(frameon=False,  bbox_to_anchor=(0.8, 1.16), ncol=1, loc='upper center', fontsize=11)

plt.xticks(rotation=0, ha='center')
plt.show()





# Define the metrics and scenarios
population_metrics = [
    'People Affected between 15 to 50 cm',
    'People Affected between 50 and 150 cm',
    'People Affected above 150 cm',
]

climate_order = ['hist', '3c', 'hightide', '3c-hightide']
adapt_order = ['hold', 'retreat']  # Specify the desired order
# Initialize an empty list to store the data for differences
diff_data = []

# Calculate differences for 'noadapt' against other adaptation scenarios
for clim_scen in climate_order:
    noadapt_values = all_infometrics_df[
        (all_infometrics_df['clim_scen'] == clim_scen) & 
        (all_infometrics_df['adapt_scen'] == 'noadapt')
    ][population_metrics].values.flatten()

    for adapt_scen in all_infometrics_df['adapt_scen'].unique():
        if adapt_scen != 'noadapt':
            adapt_values = all_infometrics_df[
                (all_infometrics_df['clim_scen'] == clim_scen) & 
                (all_infometrics_df['adapt_scen'] == adapt_scen)
            ][population_metrics].values.flatten()
            
            # Calculate the difference for each metric
            diff_values = (adapt_values - noadapt_values)/noadapt_values * 100
            diff_data.append(pd.Series([clim_scen, adapt_scen] + diff_values.tolist(), 
                                       index=['clim_scen', 'adapt_scen'] + population_metrics))

# Convert the list of Series into a DataFrame
diff_df = pd.concat(diff_data, axis=1).T

# Now, you can create a bar plot for each climate scenario with stacked bars representing the differences

# Convert the metric columns to numeric
for metric in population_metrics:
    diff_df[metric] = pd.to_numeric(diff_df[metric])

from matplotlib.patches import Patch
# Define colors for the bars, from lightest to darkest blue
blue_colors = ['#d4e6f1', '#a9cce3', '#5dade2', '#2e86c1']
hatch_patterns = {'hold': '','retreat': '///' }  # Adjust patterns as needed

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
# Add thin gray horizontal lines at intervals of 20
ax.yaxis.grid(True, zorder=0, linewidth=1.5, color='gray', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

bar_width = 0.15
positions = np.arange(len(climate_order))

# Initialize dictionaries to hold the cumulative bottoms for positive and negative stacks
pos_bottoms_dict = {adapt_scen: np.zeros(len(climate_order)) for adapt_scen in hatch_patterns.keys()}
neg_bottoms_dict = {adapt_scen: np.zeros(len(climate_order)) for adapt_scen in hatch_patterns.keys()}

# Iterate over adaptation scenarios and climate scenarios
for adapt_index, (adapt_scen, hatch) in enumerate(hatch_patterns.items()):
    # Offset to shift the bars for each adaptation scenario
    adapt_offset = (adapt_index - np.floor(len(hatch_patterns) / 2)) * bar_width

    for i, metric in enumerate(population_metrics):
        metric_values = []
        for clim_scen_index, clim_scen in enumerate(climate_order):
            value = diff_df[(diff_df['clim_scen'] == clim_scen) & (diff_df['adapt_scen'] == adapt_scen)][metric].values
            metric_value = value[0] if len(value) > 0 else 0
            metric_values.append(metric_value)
            
            # Determine whether to update the positive or negative bottom
            if metric_value > 0:
                bottom = pos_bottoms_dict[adapt_scen][clim_scen_index]
                pos_bottoms_dict[adapt_scen][clim_scen_index] += metric_value
            else:
                bottom = neg_bottoms_dict[adapt_scen][clim_scen_index]
                neg_bottoms_dict[adapt_scen][clim_scen_index] += metric_value

            # Plot the bar
            ax.bar(
                positions[clim_scen_index] + adapt_offset,
                metric_value,
                bottom=bottom,
                color=blue_colors[i],
                edgecolor='white',
                hatch=hatch,
                width=bar_width,
                zorder=3,
                label=f"{metric} ({adapt_scen})" if i == 0 and clim_scen_index == 0 else ""
            )
# Customizing the plot
# ax.set_xlabel('Climate Scenario')
ax.set_ylabel('Change in population exposure (%)')
ax.set_title('Populaton exposure wrt No-Adaptation', fontsize=14, fontweight='bold', loc='left')
ax.set_xticks(positions)
ax.set_xticklabels(climate_order)

# Create a legend for the population metrics (colors only)
color_legend_handles = [Patch(facecolor=blue_colors[i], label=population_metrics[i]) for i in range(len(population_metrics))]
# Create a legend for the adaptation scenarios (hatching patterns, using a neutral color for visibility)
hatch_legend_handles = [Patch(facecolor='grey', hatch=hatch_patterns[adapt_scen], label=adapt_scen) for adapt_scen in hatch_patterns]
# Combine both legends
legend_handles = color_legend_handles + hatch_legend_handles
# Add the combined legend to the plot
ax.legend(handles=legend_handles, frameon=False,  bbox_to_anchor=(0.8, 1.16), ncol=2, loc='upper center', fontsize=11)

plt.tight_layout()
plt.show()



# figure B
