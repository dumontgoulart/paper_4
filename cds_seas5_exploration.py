import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

file_path = 'D:/paper_4/data/seas5/ecmwf/ecmwf_eps_pf_vars_n15_s90_20170907.nc'
ds = xr.open_dataset(file_path) #.sel(time=slice('2017-09-05','2017-09-11T00:00:00.000000000'))
# convert latitude and longitude names to lat and lon
ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

print(ds)
ds['tp'] = ds['tp']*1000
ds['tp'] = ds['tp'].diff(dim='time', n=1)
ds = ds.rename({'tp':'Ptot'})

ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180


# plot a map with cartopy using the first time step of the dataset
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())    #
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.BORDERS, linestyle=':')

# plot the grid
grid = ds['Ptot'].sel(number=2).isel(time=-1)
contour = grid.plot(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap = 'Blues',robust=True, add_colorbar=True)
plt.show()

# plot timeseries of ds for all variables and for number == 1 in differnt subplots
ds['msl'].sel(number=1).mean(dim=['lat','lon']).plot()
plt.show()


# calculate ds_tp_max as the maximum of ds['tp'] along the latitude and longitude dimensions
ds_tp_max = ds['Ptot'].mean(dim=['lat','lon']).to_dataframe().reset_index()

sns.lineplot(x='time', y='Ptot', hue='number', data=ds_tp_max)
plt.show()


# Find the indices of the minimum 'msl' value over the 'lat' and 'lon' dimensions
min_msl_indices = ds['msl'].argmin(dim=['lat', 'lon'])

# Initialize lists to hold the lat, lon values
lats = []
lons = []

# Loop over all time steps and ensemble members
for t in range(len(ds['time'])):
    for n in range(len(ds['number'])):
        # Get the indices of the minimum 'msl' value for the current time step and ensemble member
        i= min_msl_indices['lat'].isel(time=t, number=n).values.tolist()
        j = min_msl_indices['lon'].isel(time=t, number=n).values.tolist()

        # Get the corresponding 'lat' and 'lon' values and append them to the lists
        lats.append(ds['lat'].isel(lat=i).values.tolist())
        lons.append(ds['lon'].isel(lon=j).values.tolist())

# Create a DataFrame
df_min_msl = pd.DataFrame({
    'time': ds['time'].values.repeat(len(ds['number'])),
    'number': ds['number'].values.tolist() * len(ds['time']),
    'lat': lats,
    'lon': lons
})


def plot_minimum_track(df):
    fig = plt.figure(figsize=(6, 8))
    central_lon = df.lon.min() + (df.lon.max() - df.lon.min())/2
    central_lat = df.lat.min() + (df.lat.max() - df.lat.min())/2
    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat))    # 
    ax.coastlines(resolution='10m') 
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.plot(df['lon'], df['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')
    # sns.lineplot(ax=ax, data = df_counter, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, sort=False, style = 'member', color = 'blue', label = 'counter') 
    plt.tight_layout()
    plt.show()
    # plt.close()
    return fig

plot_minimum_track(df_min_msl[df_min_msl['number'] == 1])



def plot_multiple_tracks(df):
    unique_numbers = df['number'].unique()
    
    fig = plt.figure(figsize=(6, 8))
    central_lon = df['lon'].min() + (df['lon'].max() - df['lon'].min()) / 2
    central_lat = df['lat'].min() + (df['lat'].max() - df['lat'].min()) / 2
    
    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat))
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    for number in unique_numbers:
        subset = df[df['number'] == number]
        ax.plot(subset['lon'], subset['lat'], transform=ccrs.PlateCarree(), linestyle='dashed', label=f'Ensemble {number}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

# Call the function with your DataFrame
fig = plot_multiple_tracks(df_min_msl)
