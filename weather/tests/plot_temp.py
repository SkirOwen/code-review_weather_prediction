import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pickle

with open("./temp", "rb") as f:
	tempe = pickle.load(f)

lons = np.arange(-180,180.25,0.25)
lats = np.arange(90,-90.25,-0.25)


plt.figure()
# to projection
ax = plt.axes(projection=ccrs.LambertCylindrical())
# form transform
ax.pcolormesh(lons, lats, tempe, transform=ccrs.PlateCarree())

ax.coastlines()

plt.show()
