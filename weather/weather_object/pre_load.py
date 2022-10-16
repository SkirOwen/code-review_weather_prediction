from weather.weather_object.weather_data import WeatherData, Group
from weather.utils.time_ops import all_timescale_from_file
from weather.utils.iterables import lst_chunk

from tqdm import tqdm


def pre_load_wds(batch_nbr, batch_size: int, cal=all_timescale_from_file()) -> None:
	cal = lst_chunk(cal, batch_size)

	for days in tqdm(cal):
		wds = Group(timescale=days)
		wds.load()
		wds.pickled()
		del wds


if __name__ == '__main__':
	pre_load_wds(batch_nbr=None, batch_size=3, cal=all_timescale_from_file()[:12])
