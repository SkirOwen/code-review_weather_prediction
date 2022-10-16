import os
import importlib
import inspect


# TODO: Check name

def get_main_dir() -> str:
    weather_module = importlib.import_module("weather")
    weather_dir = os.path.dirname(inspect.getabsfile(weather_module))
    return os.path.abspath(os.path.join(weather_dir, ".."))
