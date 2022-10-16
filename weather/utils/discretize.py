import numpy as np

from weather.utils.iterables import pairwise


def discretize_cmorph(dataset):
	cmorph_max_train = np.amax(dataset["train"]["cmorph"])
	cmorph_max_test = np.amax(dataset["test"]["cmorph"])
	cmorph_max = max(cmorph_max_test, cmorph_max_train)

	masked_train = np.ma.masked_equal(dataset["train"]["cmorph"], 0.0, copy=False)
	masked_test = np.ma.masked_equal(dataset["test"]["cmorph"], 0.0, copy=False)

	cmorph_min_train = np.amax(masked_train)
	cmorph_min_test = np.amax(masked_test)
	cmorph_min = min(cmorph_min_test, cmorph_min_train)

	del cmorph_min_test, cmorph_min_train, cmorph_max_train, cmorph_max_test

	replace_val = 0
	for val_low, val_high in pairwise((0, cmorph_min, 1, 3, 5, 10, 30, 100, cmorph_max)):
		dataset["train"]["cmorph"][
			np.logical_and(dataset["train"]["cmorph"] >= val_low, dataset["train"]["cmorph"] < val_high)] = replace_val

		dataset["test"]["cmorph"][
			np.logical_and(dataset["test"]["cmorph"] >= val_low, dataset["test"]["cmorph"] < val_high)] = replace_val

		replace_val += 1

	return dataset
