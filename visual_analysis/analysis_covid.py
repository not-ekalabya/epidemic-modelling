import folium
from folium.plugins import HeatMap

import numpy as np
import pandas as pd

import webbrowser
import os

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))

from data_covid import get_covid_data # ignore warning - intellisense issue

# consts

MAP_SAVE_LOCATION = "tmp/covid_map.html"
COUNTRY = "JP"
DATE = "2021-04-01"

map_save_location = os.path.join(
    os.getcwd(),
    MAP_SAVE_LOCATION
)


df = get_covid_data(COUNTRY)
df = df[df["date"] == DATE]

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "new_confirmed"
    ]
)


# normalize

if len(df) == 0:
    raise ValueError(
        f"No data found for {COUNTRY} on {DATE}"
    )

df["new_confirmed_norm"] = (
    df["new_confirmed"] - df["new_confirmed"].min()
) / (
    df["new_confirmed"].max() - df["new_confirmed"].min() + 1e-9
)



data = df[
    [
        "latitude",
        "longitude",
        "new_confirmed_norm"
    ]
].values.tolist()


center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()


m = folium.Map(
    [center_lat, center_lon],
    zoom_start=5
)

HeatMap(
    data,
    radius=8,
    blur=6,
    max_zoom=6
).add_to(m)


os.makedirs(
    os.path.dirname(map_save_location),
    exist_ok=True
)

m.save(map_save_location)

# open map

webbrowser.open(
    "file://" + os.path.abspath(map_save_location)
)