import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
# import cartopy
import os

# os.environ['DISPLAY'] = '172.22.224.1:0.0'
# os.environ['PROJ_LIB'] = '/users/PYS0343/rstefanescu/miniconda/envs/plot_lat_lon/share/proj/'

# Open CMORPH file
cmorph = xr.open_dataset('../../datasets/demo/CMORPH_V0.x_RAW_0.25deg-6HLY_20210403.t06z.nc')
precip = cmorph['precip'].values[0, :, :]

# Get max and min values of cmorph precipitation
print("Max and min values:", np.nanmax(precip), np.nanmin(precip))

# Get geographical coordinates
lat_CMORPH = cmorph['lat'].values
lon_CMORPH = cmorph['lon'].values

lon2d, lat2d = np.meshgrid(lon_CMORPH, lat_CMORPH)

plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
# This defines the center of the map, 180 since longitudes go from 0 to 360
clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 110, 120, 130, 140]

p1 = plt.contourf(lon2d, lat2d, precip, clevs, transform=ccrs.PlateCarree(), cmap="BuPu")
cb = plt.colorbar(p1, orientation="horizontal")
ax.coastlines()

cb.ax.set_xlabel(r" Precipitation in mm")
# plt.savefig('precip_CMORPH_grid_6hrs.png')
plt.show()
