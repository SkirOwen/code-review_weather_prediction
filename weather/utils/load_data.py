import glob
import xarray as xr
import datetime
import warnings

from typing import List, Optional

from weather.constants import *
from weather.utils.file_ops import *
from weather.utils.directories import *

import psutil

warnings.filterwarnings("default", module=__name__)


# TODO: DO NOT forget to close file after loading


def ram_batch(wds, min_ram=0.20):
	"""Give the number of batch to do given the available ram

	Parameters
	----------
	wds : list of Weather_Data,
		group of wd that needs to be load into memory
	min_ram : float, optional,
		min amount of ram that must be kept free in percentage, default set to 20%

	Returns
	-------
	batch_nbr : int,
		nbr of batch to do
	batch_size : int,
		size of each batch, batches size are as even as possible
	"""
	available_ram = psutil.virtual_memory()[1]
	data_size = 40

	if abs(data_size - available_ram) / max(data_size, available_ram) <= min_ram:
		pass


def get_cmorph_files_name(date, cycle) -> List[str]:
	"""Get the correct file name with date folder with respect to the cycle

	Parameters
	----------
	date : str,
		date to load the data, formatted as YYYYMMDD or in datetime `%Y%m%d`
	cycle : {"00", "12"},
		cycle to the data

	Returns
	-------
	list of str
	"""
	date_now = datetime.datetime.strptime(date, "%Y%m%d")
	date_tmr = date_now + datetime.timedelta(days=1)
	tmr = date_tmr.strftime("%Y%m%d")
	files_name = []

	for i in range(4):
		valid_time = CYCLE_TIME[cycle][i]
		date_finder = ((i + 9) + int(cycle)) // (12 + int(cycle))
		# generate whether to use current day or tomorrow based on the cycle
		# 00: date, date, date, tmr; 12: date, tmr, tmr, tmr
		# having a constant for that could be useful?? less computation req

		file_date = date if date_finder == 0 else tmr
		file_name = get_cmorph_filename(file_date, valid_time)

		name = os.path.join(file_date, file_name)

		files_name.append(name)
	return files_name


def load_ecmwf_xr(path, *args):
	data = xr.load_dataset(path, engine="cfgrib")
	return data


def load_ecmwf_pynio(path, fmt="grib"):
	try:
		import Nio
	except ModuleNotFoundError:
		print("Nio not install!\nPlease use load_ecmwf_xr instead")
		print("If you wish to install Nio run:\nconda install -c conda-forge pynio")
	data = Nio.open_file(path, "r", format=fmt)
	return data


def load_cmorph(path):
	data = xr.open_dataset(path)
	return data


def load_mf_cmorph(paths, parallel=True, skipna=False, keep_attrs=True, **kwargs):
	"""Open multiple files as single dataset and combining them along the time dimension.

	Parameters
	----------
	paths : str or sequences
	parallel : bool, optional
		See xr.open_mfdataset
	skipna : bool, optional
		idem
	keep_attrs : bool, optional
		idem
	**kwargs : dict, optional
		Keyword arguments to pass to xr.open_mfdataset

	Returns
	-------
	xarray.Dataset

	See Also
	--------
	xr.open_mfdataset
	"""
	# Need dask to work
	data = xr.open_mfdataset(paths, parallel=parallel, **kwargs)
	data = data.sum(dim="time", skipna=skipna, keep_attrs=keep_attrs)
	return data


def load_data(date, cycle, ecmwf_backend="xr", *args, **kwargs):
	"""Open both cmorph and ecmwf together according to the date and cycle given.

	Parameters
	----------
	date : str,
		date to load the data, formated as YYYYMMDD or in datetime `%Y%m%d`
	cycle : {"00", "12"},
		cycle to the data
	ecmwf_backend : {"pynio", "nio", "xr"}, optional
		backend to open the ecmwf, `xr` is preferred and the default
	*args : iterable, optional
		Other arguments to pass to load_ecmwf
	**kwargs : dict, optional
		Keyword arguments to pass to load_mf_cmorph

	Returns
	-------
	ndarray
	"""

	warnings.warn("Functions present here will be moved in the future", category=DeprecationWarning)

	tag = f"{date}-{cycle}"
	# could use a class to get that??

	ecmwf_path = os.path.join(get_ecmwf_dir(), date, cycle, "*")
	cmorph_paths = [os.path.join(get_cmorph_dir(), f) for f in get_cmorph_files_name(date, cycle)]

	# TODO: redo this test for backend opening of ecmwf
	ecmwf_data = None

	if ecmwf_backend == "pynio" or ecmwf_path == "nio":
		ecmwf_data = load_ecmwf_pynio(*glob.glob(ecmwf_path), *args)
	elif ecmwf_backend == "xr":
		ecmwf_data = load_ecmwf_xr(*glob.glob(ecmwf_path))

	cmorph_data = load_mf_cmorph(cmorph_paths, **kwargs)

	data = np.array([ecmwf_data, cmorph_data, tag], dtype=object)

	return data


if __name__ == '__main__':
	a = load_data("20210201", "00")
