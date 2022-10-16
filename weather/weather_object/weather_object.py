import numpy as np

from weather.constants import *
from weather.utils.time_ops import *
from weather.utils.file_ops import *


# TODO: create custom Typing for date and cycle

class WeatherObject:
    _date: str
    _cycle: str
    _tag: str
    _tmr: str

    def __init__(self, date: str, cycle: str):
        self._date = date
        self._cycle = cycle

        self._tag = self.tag
        self._tmr = self.tmr

    def __str__(self):
        return f"{self.__class__.__name__}\n{self.tag}"

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value):
        self._date = value

    @property
    def cycle(self):
        return self._cycle

    @cycle.setter
    def cycle(self, value):
        self._cycle = value

    @property
    def tag(self):
        return f"{self._date}-{self._cycle}"

    @property
    def tmr(self):
        return get_tomorrow(self.date)


# class Group(WeatherObject):
#     def __init__(self, *wobjects, **kwargs):
#         if not all([isinstance(w, WeatherObject) for w in wobjects]):
#             raise Exception("All sub-WeatherObject must be of type WeatherObject")
#         WeatherObject.__init__(self, **kwargs)
#         self.add(*wobjects)


if __name__ == '__main__':
    pass
