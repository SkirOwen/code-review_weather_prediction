import Nio
import numpy as np
import os

PATH = "../../datasets/demo"


def load_f006_file(path, display=False):
	ecmwf_forecast2 = Nio.open_file(path, "r", format="grib")

	temperature = ecmwf_forecast2.variables["2T_GDS0_SFC"][:]

	if display:
		print(ecmwf_forecast2)
		print(type(temperature), temperature.shape)  # Check type and shape of the temperature field
		print(np.min(temperature), np.max(temperature))  # Check min and max values for 2m temperature

	return ecmwf_forecast2, temperature


if __name__ == "__main__":
	file_name = 'ecmwf.t00z.pgrb.0p25.f006'
	file_path = os.path.join(PATH, file_name)
	a, b = load_f006_file(file_path, display=True)
