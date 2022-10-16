
def get_cmorph_filename(date, valid_time, time_interval="6", ext="nc") -> str:
    return f"CMORPH_V0.x_RAW_0.25deg-{time_interval}HLY_{date}.t{valid_time}z.{ext}"


def get_ecmwf_filename(cycle, ext="f024") -> str:
    return f"ecmwf.t{cycle}z.pgrb.0p25.{ext}"


if __name__ == '__main__':
    cmorph_name = get_cmorph_filename("20210206", "6")
    print(cmorph_name, "\n")
    ecmwf_name = get_ecmwf_filename("00")
    print(ecmwf_name)
