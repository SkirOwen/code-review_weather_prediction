import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean

from weather.utils.load_data import *


date = str(input("Date (YYYYMMDD): "))
cycle = str(input("Cycle: "))

print("Plotting....")

cmorp = load_data(date, cycle, ecmwf_backend="nio", skipna=True)[1]

precip = cmorp["precip"].values[:, :]
lat_cmorph, lon_cmorph = cmorp["lat"].values, cmorp["lon"].values
lon2d, lat2d = np.meshgrid(lon_cmorph, lat_cmorph)

plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 110, 120, 130, 140]
# clevs = [x for x in range(int(np.nanmax(precip)))]

p1 = plt.contourf(lon2d, lat2d, precip, clevs, transform=ccrs.PlateCarree(), cmap=cmocean.cm.rain)
cb = plt.colorbar(p1, orientation="horizontal")
ax.coastlines()

cb.ax.set_xlabel(r"Precipitation in $mm$" + "\n" + date + " | " + cycle)
plt.show()
