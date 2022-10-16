import numpy as np
import xarray as xr
import os

PATH = "../../datasets/demo"


def load_nc_file(path, display=False):
	precip_6hrs_accumulation = xr.open_dataset('CMORPH_V0.x_RAW_0.25deg-6HLY_20210403.t06z.nc')
	precip = precip_6hrs_accumulation['precip'].values  # Extract 6hrs total precipitation accumulation

	if display:
		print(type(precip), precip.shape)  # Check type and shape of precip

		# Print statistics - please be aware that precipitation is available only from 60S to 60N latitudes
		# so NaNs are present outside this latitude band
		print("max value of 6hr precip accumulation:", np.nanmax(precip[0, :, :]))
		print("min value of 6hr precip accumulation:", np.nanmin(precip[0, :, :]))

	return precip_6hrs_accumulation, precip


if __name__ == '__main__':
	file_name = 'CMORPH_V0.x_RAW_0.25deg-6HLY_20210403.t06z.nc'
	file_path = os.path.join(PATH, file_name)
	a, b = load_nc_file(file_path, display=True)


