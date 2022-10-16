import gzip
import os
import pickle
import gc
from typing import Dict
from tqdm import tqdm
import h5py

import numpy as np
import torch
import torch.utils.data as data_utils

from weather.weather_object.weather_data import WeatherData, Group
from weather.utils.directories import get_pickle_dir, get_h5_dir
from weather.utils.time_ops import all_timescale_from_file
from weather.utils.discretize import discretize_cmorph


def prep_data_load() -> None:
	g = Group()
	gc.disable()
	g.pickled()
	gc.enable()


def prep_data_load_month() -> None:
	cal = all_timescale_from_file()[:31] + all_timescale_from_file()[-31:]
	g = Group(timescale=cal)
	gc.disable()
	g.pickled(step="dec-march")
	gc.enable()


def gen_data(filename="wds-1201-0331-1", test_size=62):
	path = os.path.join(get_pickle_dir(), filename)

	if not os.path.exists(path):
		print("No wds present, loading and pickling ALL the wds. Please wait")
		prep_data_load()

	with gzip.open(path, "rb") as f:
		print("Loading the pickle")
		# wds: List
		wds = pickle.load(f)

	num_data = len(wds)

	cmorph = np.zeros((num_data, 721, 1440))
	ecmwf = np.zeros((num_data, 8, 721, 1440))
	for i, wd in enumerate(wds):
		print(f"\rProcessing wd #{i}", end="")
		a_cmorph, a_ecmwf, lat2d, lon2d = wd.to_numpy()
		ecmwf_lat_lon = np.vstack((a_ecmwf, lat2d[None, :, :], lon2d[None, :, :]))

		# print(ecmwf_lat_lon.shape)

		cmorph[i] = a_cmorph
		ecmwf[i] = ecmwf_lat_lon
	# print(ecmwf.shape)
	print("out of loop")
	del wds

	print("test")
	if test_size is None:
		weather_train = {"ecmwf": ecmwf, "cmorph": cmorph}
		weather_test = {"ecmwf": np.array([]), "cmorph": np.array([])}
	else:
		weather_train = {"ecmwf": ecmwf[:-test_size], "cmorph": cmorph[:-test_size]}
		weather_test = {"ecmwf": ecmwf[-test_size:], "cmorph": cmorph[-test_size:]}

	datasets = {"train": weather_train, "test": weather_test}
	print("removing cmorph and ecmwf array")
	del cmorph, ecmwf

	datasets_path = os.path.join(get_pickle_dir(), f"dict_{filename}")

	print("Pickling")
	with gzip.open(datasets_path, "wb") as f:
		gc.disable()
		pickle.dump(datasets, f, protocol=-1)
		gc.enable()
	print("Ding!")


def gen_h5_data_pickle(filename="wds-1201-0331-1", test_size=62):
	path = os.path.join(get_pickle_dir(), filename)

	with gzip.open(path, "rb") as f:
		print("Loading the pickle")
		# wds: List
		wds = pickle.load(f)

	num_data = len(wds)
	cmorph = np.zeros((num_data, 1, 721, 1440))
	ecmwf = np.zeros((num_data, 8, 721, 1440))
	for i, wd in enumerate(wds):
		print(f"\rProcessing wd #{i}", end="")
		a_cmorph, a_ecmwf, lat2d, lon2d = wd.to_numpy()
		ecmwf_lat_lon = np.vstack((a_ecmwf, lat2d[None, :, :], lon2d[None, :, :]))

		# print(ecmwf_lat_lon.shape)

		cmorph[i] = a_cmorph
		ecmwf[i] = ecmwf_lat_lon
	# print(ecmwf.shape)
	print("out of loop")
	del wds

	datasets_path = os.path.join(get_h5_dir(), f"dict_{filename}.hdf5")

	with h5py.File(datasets_path, "w") as f:
		train_grp = f.create_group("train")
		test_grp = f.create_group("test")

		train_ecmwf = train_grp.create_dataset("ecmwf", data=ecmwf[:-test_size])
		test_ecmwf = test_grp.create_dataset("ecmwf", data=ecmwf[-test_size:])

		train_ecmwf = train_grp.create_dataset("cmorph", data=cmorph[:-test_size])
		test_ecmwf = test_grp.create_dataset("cmorph", data=cmorph[-test_size:])

	print("done!")


def gen_load_data(batch_size, dataset_file="wds-1201-0331-dec-march", test_size=62, discretize=False):
	path = os.path.join(get_pickle_dir(), dataset_file)

	if not os.path.exists(path):
		print("No wds present, loading and pickling ALL the wds. Please wait")
		prep_data_load_month()

	with gzip.open(path, "rb") as f:
		print("Loading the pickle")
		# wds: List
		wds = pickle.load(f)

	num_data = len(wds)

	cmorph = np.zeros((num_data, 721, 1440))
	ecmwf = np.zeros((num_data, 8, 721, 1440))
	for i, wd in enumerate(wds):
		print(f"\rProcessing wd #{i}", end="")
		a_cmorph, a_ecmwf, lat2d, lon2d = wd.to_numpy()
		ecmwf_lat_lon = np.vstack((a_ecmwf, lat2d[None, :, :], lon2d[None, :, :]))

		cmorph[i] = a_cmorph
		ecmwf[i] = ecmwf_lat_lon
	print("out of loop")
	del wds

	print("test")
	weather_train = {"ecmwf": ecmwf[:-test_size], "cmorph": cmorph[:-test_size]}
	weather_test = {"ecmwf": ecmwf[-test_size:], "cmorph": cmorph[-test_size:]}

	datasets = {"train": weather_train, "test": weather_test}
	print("removing cmorph and ecmwf array")
	del cmorph, ecmwf

	if discretize:
		datasets = discretize_cmorph(datasets)

	train_data = torch.from_numpy(
		datasets["train"]["ecmwf"][:, :, 121:600, :].astype(np.float32))
	train_truths = torch.from_numpy(
		datasets["train"]["cmorph"][:, None, 121:600, :].astype(np.float32))

	train_dataset = data_utils.TensorDataset(train_data, train_truths)
	train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	test_data = torch.from_numpy(
		datasets["test"]["ecmwf"][:, :, 121:600, :].astype(np.float32))
	test_truths = torch.from_numpy(
		datasets["test"]["cmorph"][:, None, 121:600, :].astype(np.float32))

	test_dataset = data_utils.TensorDataset(test_data, test_truths)
	test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	return train_loader, test_loader, train_dataset, test_dataset


def load_data(batch_size, dataset_file="dict_wds-1201-0331-1", discretize=False):
	path = os.path.join(get_pickle_dir(), dataset_file)

	if not os.path.exists(path):
		print("File did not exist, calling gen_data with default args")
		gen_data()

	with gzip.open(path, 'rb') as f:
		print("Loading the pickle")
		dataset = pickle.load(f)

	if discretize:
		dataset = discretize_cmorph(dataset)

	train_data = torch.from_numpy(
		dataset["train"]["ecmwf"][:, :, 121:600, :].astype(np.float32))
	train_truths = torch.from_numpy(
		dataset["train"]["cmorph"][:, None, 121:600, :].astype(np.float32) * 1e3)

	train_dataset = data_utils.TensorDataset(train_data, train_truths)
	train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,
	                                     pin_memory=True)

	test_data = torch.from_numpy(
		dataset["test"]["ecmwf"][:, :, 121:600, :].astype(np.float32))
	test_truths = torch.from_numpy(
		dataset["test"]["cmorph"][:, None, 121:600, :].astype(np.float32) * 1e3)

	# Maybe drop last
	test_dataset = data_utils.TensorDataset(test_data, test_truths)
	test_loader = data_utils.DataLoader(test_dataset, drop_last=True, batch_size=batch_size, num_workers=8,
	                                    shuffle=True, pin_memory=True)

	return train_loader, test_loader, train_dataset, test_dataset


def load_h5(batch_size, dataset_file="dict_wds-1201-0331-1.hdf5", discretize=False):
	"""dataset_file MUST have the extension hdf5 included!!"""

	filepath = os.path.join(get_h5_dir(), dataset_file)

	with h5py.File(filepath, "r") as f:
		train_data = torch.from_numpy(
			f["train"]["ecmwf"][:, :, 121:600, :].astype(np.float32))
		train_truths = torch.from_numpy(
			f["train"]["cmorph"][:, :, 121:600, :].astype(np.float32) * 1000)

		train_dataset = data_utils.TensorDataset(train_data, train_truths)
		train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,
												pin_memory=True)

		test_data = torch.from_numpy(
			f["test"]["ecmwf"][:, :, 121:600, :].astype(np.float32))
		test_truths = torch.from_numpy(
			f["test"]["cmorph"][:, :, 121:600, :].astype(np.float32) * 1000)

	test_dataset = data_utils.TensorDataset(test_data, test_truths)
	test_loader = data_utils.DataLoader(test_dataset, drop_last=True, batch_size=batch_size, num_workers=8,
										shuffle=True, pin_memory=True)

	return train_loader, test_loader, train_dataset, test_dataset


def main():
	# prep_data_load_month()
	prep_data_load()
	gen_h5_data_pickle()
	import time
	start = time.time()
	train_loader_h5, test_loader_h5, train_dataset_h5, test_dataset_h5 = load_h5(1)
	stop = time.time()
	# print(stop - start)
	# train_loader, test_loader, train_dataset, test_dataset = load_data(1)

# start = time.time()
# gen_data("wds-1201-0331-dec-march")
# stop = time.time()
# print(stop - start)
# # gen_load_data(14)
# load_data(1)


if __name__ == '__main__':
	main()
