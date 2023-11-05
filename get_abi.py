from pathlib import Path
import numpy as np
from datetime import datetime
from ezgoes import GetGOES, GOES_Product

from krttdkit.products import ABIL1b
from krttdkit.operate.geo_helpers import get_geo_range
from krttdkit.operate.recipe_book import abi_recipes
from krttdkit.visualize import geoplot as gp

def get_goes_fg(target_time, data_dir):
    nc_files = ABIL1b.get_l1b(
        data_dir=data_dir,
        satellite="17",
        scan="C",
        bands=(1,2,3,4,5,7,11,13,14,15),
        start_time=target_time,
        replace=False,
        )[0]
    fg = ABIL1b.from_l1b_files(
            path_list=nc_files,
            convert_tb=True,
            convert_ref=True,
            get_latlon=True,
            get_mask=True,
            get_scanangle=False,
            resolution=1,
            )

    latlon = np.dstack((fg.data("lat"), fg.data("lon")))
    yrange,xrange = get_geo_range(latlon, (37.5,-127.5), 1024, 1024,
                                  True, True, False)
    fg = fg.subgrid(vrange=yrange, hrange=xrange)
    fg.to_pkl(data_dir.joinpath(
        target_time.strftime("abil1b_%Y%m%d-%H%M.pkl")))

if __name__=="__main__":
    times = [
            ## Terra times
            datetime(2021,8,16,19,25),
            datetime(2021,8,18,19,12),
            datetime(2021,8,20,19,0),
            datetime(2021,8,22,18,48),
            datetime(2021,8,23,19,31),
            datetime(2021,8,25,19,18),

            ## Aqua times
            datetime(2021,8,16,21,7),
            datetime(2021,8,17,21,50),
            datetime(2021,8,19,21,37),
            datetime(2021,8,21,21,25),
            datetime(2021,8,23,21,13),
            ]
    data_dir = Path("data/abi")
    fig_dir = Path("figures/abi")

    #for t in times: get_goes_fg(t, data_dir)

    fg_paths = [p for p in data_dir.iterdir() if "abil1b" in p.name]
    for p in fg_paths:
        fg = ABIL1b.from_pkl(p)
        gp.generate_raw_image(fg.data("truecolor"), fig_dir.joinpath(
            p.name.replace("abil1b", "abi_tc").replace("pkl", "png")))
        gp.generate_raw_image(fg.data("dcp"), fig_dir.joinpath(
            p.name.replace("abil1b", "abi_dcp").replace("pkl", "png")))
