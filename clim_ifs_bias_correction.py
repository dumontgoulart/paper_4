# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:04:50 2023

@author: H. Goulart
"""
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np

import metpy.calc as mpcalc
import metpy.xarray as mpxarray
from scipy.signal import savgol_filter
from scipy.interpolate import BSpline
from scipy.ndimage import laplace

import clim_functions as climfun 
################################################################################

def bias_correct_dataset(ds, ds_composite_gpm, ds_composite_ens, ds_storm_subset_1hour, 
                         bc_start_date, bc_end_date, output_path=None):
    """
    Applies bias correction to a dataset and optionally saves the result as a NetCDF file.

    Parameters:
    ds (xarray.Dataset): Dataset to be bias-corrected.
    ds_composite_gpm (xarray.Dataset): GPM dataset used for bias correction.
    ds_composite_ens (xarray.Dataset): Ensemble dataset used for bias correction.
    ds_storm_subset_1hour (xarray.Dataset): Hourly storm dataset.
    bc_start_date (str): Start date for the bias correction period.
    bc_end_date (str): End date for the bias correction period.
    output_path (str, optional): File path to save the bias-corrected dataset as a NetCDF file.

    Returns:
    xarray.Dataset: Bias-corrected dataset.
    """
    if 'tprate' in ds_composite_ens:
        print('converting tprate to tp')
    # Calculate bias correction ratios
    bc_ratio_tp = (ds_composite_gpm['tp'].mean(dim=['lat','lon'])
                   .sel(time=slice(bc_start_date, bc_end_date)).mean() / 
                   ds_composite_ens['tprate'].mean(dim=['lat','lon']).mean(dim=['number'])
                   .sel(time=slice(bc_start_date, bc_end_date)).mean())

    bc_ratio_msl = (ds_storm_subset_1hour['msl'].sel(time=slice(bc_start_date, bc_end_date)).mean() / 
                    ds_composite_ens['msl'].min(dim=['lat','lon']).mean(dim=['number'])
                    .sel(time=slice(bc_start_date, bc_end_date)).mean())

    bc_ratio_U10 = (ds_storm_subset_1hour['U10'].sel(time=slice(bc_start_date, bc_end_date)).mean() / 
                    ds_composite_ens['U10'].max(dim=['lat','lon']).mean(dim=['number'])
                    .sel(time=slice(bc_start_date, bc_end_date)).mean())

    # Apply bias correction
    ds_ens_bc = ds.copy()
    ds_ens_bc['tp'] = ds['tprate'] * bc_ratio_tp
    ds_ens_bc['msl'] = ds['msl'] * bc_ratio_msl
    ds_ens_bc['U10'] = ds['U10'] * bc_ratio_U10

    # Update metadata
    ds_ens_bc['tp'].attrs.update({'units': 'mm/h', 'long_name': 'Total precipitation rate'})
    ds_ens_bc['msl'].attrs.update({'units': 'hPa', 'long_name': 'Mean sea level pressure'})
    ds_ens_bc['U10'].attrs.update({'units': 'm/s', 'long_name': '10 metre total wind'})

    # Adjust wind components based on the original wind direction
    ds_ens_bc['wind_direction'] = climfun.calculate_wind_direction_xarray(ds, u10_var='u10', v10_var='v10')
    ds_ens_bc['u10'], ds_ens_bc['v10'] = climfun.decompose_wind(ds_ens_bc, U10_var='U10', wind_dir_var='wind_direction')

    # Save as NetCDF if output path is provided
    if output_path:
        ds_ens_bc.to_netcdf(output_path)

    return ds_ens_bc

# Load data and pararmeters
file_path = 'D:/paper_4/data/seas5/ecmwf/ecmwf_eps_pf_010_vars_n50_s60_20190313.nc' # ecmwf_eps_pf_vars_n15_s90_20190313 / ecmwf_eps_pf_vars_n15_s90_20170907
start_time = '2019-03-14T00:00:00.000000000'
end_time = '2019-03-15T12:00:00.000000000'

# load IFS ensemble
ds = xr.open_dataset(file_path).sel(time=slice(start_time,end_time))
ds = climfun.preprocess_era5_dataset(ds, forecast=True)
ds['tprate'] = ds['tprate']*3.6*1000

# load ERA5 dataset
ds_era5 = xr.open_dataset(r'D:\paper_4\data\era5\era5_hourly_vars_idai_single_2019_03.nc').sel(time=slice(start_time,end_time)).rename({'latitude': 'lat', 'longitude': 'lon'})
ds_era5 = climfun.preprocess_era5_dataset(ds_era5)

ds_control = xr.open_dataset(r'D:\paper_4\data\seas5\ecmwf\ecmwf_eps_cf_010_vars_s90_20190313_00.nc').sel(time=slice(start_time,end_time))
ds_control = climfun.preprocess_era5_dataset(ds_control, forecast=True)

# Load GPM dataset
ds_gpm = xr.open_dataset(r'D:\paper_4\data\nasa_data\gpm_imerg_201903.nc').sel(time=slice(start_time,end_time))
ds_gpm.coords['lon'] = (ds_gpm.coords['lon'] + 180) % 360 - 180
ds_gpm = ds_gpm.sortby('lat', ascending=False)
ds_gpm = ds_gpm.rename({'precipitation': 'tp'})

ds_gpm_hour = ds_gpm.resample(time='1H').mean()
# save ds_gpm_hour to netcdf
# ds_gpm_hour.to_netcdf(r'D:\paper_4\data\nasa_data\gpm_imerg_201903_hourly.nc')

# plot timeseries average ds_gpm and ds_gpm_hour
ds_gpm['tp'].mean(dim=['lat','lon']).plot(label='gpm')
ds_gpm_hour['tp'].mean(dim=['lat','lon']).plot(label='gpm_hour')
plt.legend()
plt.show()

# Load IBTrACS dataset
df_storm, df_storm_subset = climfun.process_ibtracs_storm_data(ibtracs_path = r'D:\paper_4\data\ibtracs\IBTrACS.since1980.v04r00.nc',
                                                  storm_id = '2019063S18038', 
                                                  ds_time_range = ds)
ds_storm_subset_1hour = climfun.interpolate_dataframe(df_storm_subset, ['U10', 'msl', 'time', 'lat', 'lon'], 'time', '1H')

# multiply tprate by 3600 to get mm/h
ds['tp'].mean(dim=['lat','lon', 'number']).plot(label='tp')
ds['tprate'].mean(dim=['lat','lon', 'number']).plot(label='tprate')
ds_gpm['tp'].mean(dim=['lat','lon']).plot(label='gpm')
ds_era5['tp'].mean(dim=['lat','lon']).plot(label='era5')
plt.legend()
plt.show()


# track storm 
storm_track_ens = climfun.storm_tracker_mslp_ens(ds, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)
# track ds_era5 storm
storm_track_era5 = climfun.storm_tracker_mslp_updated(ds_era5, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)
# track control run
storm_track_control = climfun.storm_tracker_mslp_updated(ds_control, df_storm, smooth_step='savgol', large_box=3, small_box=1.5)


# plot storm tracks 
climfun.plot_minimum_track(storm_track_ens)
climfun.plot_minimum_track(storm_track_era5, df_storm_subset)
climfun.plot_minimum_track(storm_track_control, df_storm_subset)

# Calculate composite
ds_composite_ens = climfun.storm_composite_ens(ds, storm_track_ens, radius = 2)
ds_composite_era5 = climfun.storm_composite(ds_era5, storm_track_era5, radius = 2)
ds_composite_control = climfun.storm_composite(ds_control, storm_track_control, radius = 2)
# use the observed track to calculate the composite for gpm
ds_composite_gpm = climfun.storm_composite(ds_gpm_hour, ds_storm_subset_1hour.to_dataframe(), radius = 2)

# BIAS CORRECTION ##############################################################
bc_start_date = '2019-03-14T18:00:00.000000000'
bc_end_date = '2019-03-15T06:00:00.000000000'
path_bc_file = r'D:\paper_4\data\seas5\bias_corrected\ecmwf_eps_pf_010_vars_n50_s60_20190313_bc.nc'
# Example usage
ds_ens_bc = bias_correct_dataset(ds, ds_composite_gpm, ds_composite_ens, ds_storm_subset_1hour,
                                 bc_start_date, bc_end_date, path_bc_file)

# CHECK RESULTS ################################################################
ds_composite_ens_bc = climfun.storm_composite_ens(ds_ens_bc, storm_track_ens, radius = 2)

# NON BIAS corrected
climfun.plot_comp_variable_timeseries(ds_composite_ens, 'tp', ds_era5_composite=ds_composite_gpm, agg_method='mean')
climfun.plot_comp_variable_timeseries(ds_composite_ens, 'U10', ds_era5_composite=ds_composite_era5, agg_method='max', obs_data=df_storm_subset)
climfun.plot_comp_variable_timeseries(ds_composite_ens, 'msl', ds_era5_composite=ds_composite_era5, agg_method='min', obs_data=df_storm_subset)

# plot timeseries average for bc u10 and msl
climfun.plot_comp_variable_timeseries(ds_composite_ens_bc, 'U10', ds_era5_composite=ds_composite_era5, agg_method='max', obs_data=df_storm_subset)
climfun.plot_comp_variable_timeseries(ds_composite_ens_bc, 'msl', ds_era5_composite=ds_composite_era5, agg_method='min', obs_data=df_storm_subset)
climfun.plot_comp_variable_timeseries(ds_composite_ens_bc, 'tp', ds_era5_composite=ds_composite_gpm, agg_method='mean')

# plot ds_composite_ens_bc and ds_composite_gpm and ds_composite_ens
ds_composite_ens['tp'].mean(dim=['lat','lon','number']).plot(label='ensemble raw')
ds_composite_ens_bc['tp'].mean(dim=['lat','lon','number']).plot(label='ensemble bias corrected')
ds_composite_gpm['tp'].mean(dim=['lat','lon']).plot(label='gpm (ref)')
plt.axvspan(bc_start_date, bc_end_date, color='grey', alpha=0.2)
plt.legend()
plt.title('Mean precipitation rate over time with bias correction')
plt.show()

