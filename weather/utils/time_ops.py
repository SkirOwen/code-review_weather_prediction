import datetime
import os

from weather.utils.directories import get_cmorph_dir, get_ecmwf_dir
from weather.utils.iterables import sorted_alphanumeric
from typing import List


def get_tomorrow(date: str) -> str:
    date_tmr = datetime.datetime.strptime(date, "%Y%m%d") + datetime.timedelta(days=1)
    tmr = date_tmr.strftime("%Y%m%d")
    return tmr


def all_timescale_from_file(**kwargs) -> List[str]:
    # TODO: what if not the same number of files in cmorph and ecmwf

    cmorph_dates = sorted_alphanumeric(os.listdir(get_cmorph_dir()))
    ecmwf_dates = sorted_alphanumeric(os.listdir(get_ecmwf_dir()))

    cmorph_len = len(cmorph_dates)
    ecmwf_len = len(ecmwf_dates)

    if ecmwf_len != cmorph_len:
        pass

    calendar = [x for x in cmorph_dates if x in ecmwf_dates]
    return calendar


if __name__ == '__main__':
    d = str(input("Date (YYYYMMDD): "))
    print(get_tomorrow(d))
