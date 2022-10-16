import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean

from weather.utils.load_data import *


def var_plot(data, cmap=cmocean.cm.rain, save=False) -> None:
    lat_cmorph, lon_cmorph = cmorph["lat"].values, cmorph["lon"].values
    lon2d, lat2d = np.meshgrid(lon_cmorph, lat_cmorph)

    plt.figure(figsize=(20, 10))
    # TODO: change the longitude
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 110, 120, 130, 140]

    # clevs = [x for x in range(int(np.nanmax(precip)))]

    p1 = plt.contourf(lon2d, lat2d, precip, clevs, transform=ccrs.PlateCarree(), cmap=cmap)
    cb = plt.colorbar(p1, orientation="horizontal")
    ax.coastlines()

    cb.ax.set_xlabel(r"Precipitation in $mm$" + "\n" + date + " | " + cycle)
    if save:
        plt.savefig()
        print("saved")
    plt.show()


def plot_3d_grid(grid):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


if __name__ == '__main__':
    from weather.weather_object.weather_data import WeatherData
    date = str(input("Date (YYYYMMDD): "))
    cycle = str(input("Cycle: "))

    wd = WeatherData(date, cycle)
    wd.load_data()
    cmorph = wd.cmorph_data
    precip = cmorph["precip"].values

    var_plot(precip)

