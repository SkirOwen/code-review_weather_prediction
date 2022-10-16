import os

import h5py
import numpy as np
import xarray as xr
import pickle
import gzip

from tqdm import tqdm
from typing import Tuple

import weather
from weather.weather_object.weather_object import WeatherObject
from weather.constants import *
from weather.utils.time_ops import *
from weather.utils.file_ops import *
from weather.utils.directories import *


class WeatherData(WeatherObject):
	def __init__(self, date: str, cycle: str):
		super().__init__(date, cycle)

		self.filepaths = [[], []]

		# check this if not other class
		self.cmorph_data = None
		self.ecmwf_data = None

		self.init_filepaths()

	def init_filepaths(self):
		# FIXME: regen file path when change date
		self.get_cmorph_filepaths()
		self.get_ecmwf_filepath()

	def get_cmorph_filepaths(self):
		# TODO: for range(4) -> to range(CONST)
		cmorph_filepaths = []
		for i in range(4):
			valid_time = CYCLE_TIME[self.cycle][i]
			date_finder = DATE_FINDER_LOOKUP[self.cycle][i]

			file_date = self.date if date_finder == 0 else self.tmr
			file_name = get_cmorph_filename(file_date, valid_time)

			cmorph_filepaths.append(os.path.join(get_cmorph_dir(), file_date, file_name))
		self.filepaths[1] = cmorph_filepaths

	def get_ecmwf_filepath(self):
		ecmwf_filepath = [os.path.join(get_ecmwf_dir(), self.date, self.cycle, get_ecmwf_filename(self.cycle))]
		self.filepaths[0] = ecmwf_filepath

	# Data loading
	def load_data(self, drop_isobaric: bool = True, fillnan_zero: bool = True, backend: str = "cfgrib"):
		self.load_cmorph(skipna=fillnan_zero)
		self.load_ecmwf(drop_isobaric, fillnan_zero, backend)

	def load_cmorph(self, skipna: bool = False, keep_attrs: bool = True):
		data = xr.open_mfdataset(self.filepaths[1], parallel=True)
		self.cmorph_data = data.sum(dim="time", skipna=skipna, keep_attrs=keep_attrs)
		return self

	def load_ecmwf(self, drop_isobaric: bool, fillnan_zero: bool, backend: str = "cfgrib"):
		if backend == "cfgrib":
			data = xr.load_dataset(*self.filepaths[0], engine="cfgrib")
			self.ecmwf_data = data.drop_vars(ECMWF_DROP_VAR)
			if fillnan_zero:
				self.ecmwf_data = self.ecmwf_data.fillna(0)
			if drop_isobaric:
				self.ecmwf_data = self.ecmwf_data.drop_dims("isobaricInhPa")

		else:
			print("To use Pynio, use the function in load_data.py"
					" and pass the self.filepaths[0] for the files")

		return self

	def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		# Make sure data is present, no matter
		if self.ecmwf_data is None and self.cmorph_data is None:
			self.load_data()
		elif self.cmorph_data is None:
			self.load_cmorph()
		elif self.ecmwf_data is None:
			self.load_ecmwf(True, True)

		# xarray.Dataset
		data = self.ecmwf_data

		# TODO: get the size of a through func
		a_ecmwf = np.empty([len(data.data_vars), data.dims["latitude"], data.dims["longitude"]], dtype=np.float32)
		for i, var in enumerate(["v10", "tp", "t2m", "d2m", "u10", "tcw"]):
			a_ecmwf[i] = data[var].values.astype(np.float32)

		# the 1e-3 is to change the unit of the cmorph from mm to m
		a_cmorph = self.cmorph_data["precip"].values.astype(np.float32) * 1e-3

		# Roll cmorph (any size) of phi=180deg so central longitude be 0
		a_cmorph = np.roll(a_cmorph, a_cmorph.shape[1]//2, 1)

		a_cmorph = np.array([a_cmorph])

		lat = self.cmorph_data["lat"].values
		lon = self.ecmwf_data["longitude"].values

		lon2d, lat2d = np.meshgrid(lon, lat)

		return a_cmorph, a_ecmwf, lat2d, lon2d

	def close_files(self) -> None:
		for cmorph_files, ecmwf_files in zip(self.cmorph_data, self.ecmwf_data):
			cmorph_files.close()
			ecmwf_files.close()

	def pickled(self, filename=None) -> None:
		filename = filename or self.tag
		with open(os.path.join(get_pickle_dir(), filename), "wb") as f:
			pickle.dump(self, f, protocol=-1)


class Group:
	def __init__(self, timescale=all_timescale_from_file()):
		self.wds = []
		self.timescale = timescale

		self.init_load_wd()

	def __len__(self):
		return len(self.wds)

	# TODO:  redo the tqdm for loading time
	def init_load_wd(self):
		self.init_wd()
		self.load()

	def init_wd(self):
		for date in tqdm(self.timescale, desc=f"Initialise"):
			for cycle in ["00", "12"]:
				wd = weather.WeatherData(date, cycle)
				self.wds.append(wd)

	def load(self):
		for wd in tqdm(self.wds, desc=f"Load"):
			wd.load_data()

	def pickled(self, step=None):
		cal = self.timescale
		step = int(cal[1])-int(cal[0]) if step is None else step
		filename = f"wds-{cal[0][-4:]}-{cal[-1][-4:]}-{step}"
		# name struct: wds-STR_DATE-END_DATE-STEP
		with gzip.open(os.path.join(get_pickle_dir(), filename), "wb") as p_file:
			pickle.dump(np.array(self.wds), p_file, protocol=-1)
			print(f"the wds list of {self.__class__.__name__} is pickled at {p_file}")
			# TODO : change this msg

	def h5(self, step=None):
		cal = self.timescale
		step = int(cal[1])-int(cal[0]) if step is None else step
		filename = f"wds-{cal[0][-4:]}-{cal[-1][-4:]}-{step}.hdf5"

		with h5py.File(os.path.join(get_h5_dir(), filename), "w") as f:
			for wd in tqdm(self.wds, desc=f"h5py"):
				a_cmorph, a_ecmwf, lat2d, lon2d = wd.to_numpy()
				ecmwf_aug = np.vstack((a_ecmwf, lat2d[None, :, :], lon2d[None, :, :]))

				tag_grp = f.create_group(f"{wd.tag}")
				cmorph_dset = tag_grp.create_dataset("cmorph", data=a_cmorph)
				ecmwf_dset = tag_grp.create_dataset("ecmwf", data=ecmwf_aug)
		print("done!")


# Maybe do this in a class?
def group(timescale: List = all_timescale_from_file()) -> list:
	wds = []
	for date in tqdm(timescale):
		for cycle in ["00", "12"]:
			wd = weather.WeatherData(date, cycle)
			wd.load_data()
			wds.append(wd)
	return wds


def load_group(group) -> None:
	for wd in group:
		wd.load_data()


def main():
	cal = all_timescale_from_file()

	wds = Group(cal)
	wds.load()
	from time import time
	start = time()
	wds.h5()
	stop = time()
	print(stop - start)


if __name__ == '__main__':
	main()
	# wd = weather.WeatherData("20210109", "00")
	# wd.load_data()
	# cm, ec, la, lo = wd.to_numpy()
