import numpy as np


CYCLE_TIME = {
	"00": ["06", "12", "18", "00"],
	"12": ["18", "00", "06", "12"]
}

DATE_FINDER_LOOKUP = {
	"00": [0, 0, 0, 1],
	"12": [0, 1, 1, 1]
}

# PROJECTION_CENTRAL_LONGITUDE
CMORPH_CENT_LON = 180
ECMWF_CENT_LON = 0
#
# VARIABLES_ECMWF = {
#     "Z_GDS0_ISBL": "geopotential",
#     "T_GDS0_ISBL": "temperature",
#     "U_GDS0_ISBL": "u_velocity",
#     "V_GDS0_ISBL": "v_velocity",
#     "SP_GDS0_SFC": "surface_pressure",
#     "TCW_GDS0_SFC": "total_column_water",
#     "MSL_GDS0_SFC": "mean_sea_pressure",
#     "R_GDS0_ISBL": "relative_humidity",
#     "10U_GDS0_SFC": "",
#     "10V_GDS0_SFC": "",
#     "2T_GDS0_SFC": "",
#     "2D_GDS0_SFC": "",
#     "TP_GDS0_SFC": "",
#     "g0_lon_2": "",
#     "lv_ISBL0": "",
# }

# All var in ECMWF expect tp (total preci)
ECMWF_DROP_VAR = [
	"z",
	"t",
	"u",
	"v",
	"sp",
	# "tcw",
	"msl",
	"r",
	# "u10",
	# "v10",
	# "t2m",
	# "d2m",
]

